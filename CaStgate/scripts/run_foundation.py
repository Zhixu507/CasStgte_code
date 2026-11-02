"""Frozen linear probes for CTransPath and UNI.

Five patient folds and seeds 42, 43, 44. The encoder stays frozen. The linear
head uses the same augmentation, learning rate, weight decay, epoch budget,
and validation Macro-F1 selection as the single-branch models. Test images
are scored only after that selection.
"""

import argparse
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = {"ctranspath": ROOT / "ctranspath.pth", "uni": ROOT / "pytorch_model.bin"}
SEEDS = (42, 43, 44)
FOLDS = (0, 1, 2, 3, 4)


def config_for(name, fold, seed):
    output = ROOT / "runs" / "foundation" / name / f"fold{fold}_seed{seed}"
    return {
        "manifest": "data/lunghist700/manifest.csv",
        "splits": "data/lunghist700/splits.json",
        "output": str(output),
        "fold": fold,
        "seed": seed,
        "image_size": 224,
        "batch_size": 8,
        "num_workers": 0,
        "epochs": 40,
        "patience": 10,
        "lr": 1e-4,
        "weight_decay": 0.01,
        "device": "cuda",
        "cpu_threads": 4,
        "foundation": name,
        "weights": str(WEIGHTS[name]),
        "encoder_frozen": True,
        "num_stain_views": 2,
        "stain_strength": 0.15,
        "bootstrap_samples": 1000,
    }


def run_one(name, fold, seed):
    from castgate.foundation import train_probe
    os.chdir(ROOT)
    train_probe(config_for(name, fold, seed))


def pending_jobs():
    jobs = []
    for name in ("ctranspath", "uni"):
        if not WEIGHTS[name].is_file():
            raise FileNotFoundError(WEIGHTS[name])
        for fold in FOLDS:
            for seed in SEEDS:
                output = ROOT / "runs" / "foundation" / name / f"fold{fold}_seed{seed}"
                if (output / "test" / "metrics.json").exists():
                    continue
                if output.exists():
                    shutil.rmtree(output)
                jobs.append((name, fold, seed))
    return jobs


def run_parallel(jobs, gpus):
    pending = queue.Queue()
    for job in jobs:
        pending.put(job)
    failed, done, lock = [], [], threading.Lock()
    python = sys.executable

    def worker(gpu):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), CUBLAS_WORKSPACE_CONFIG=":4096:8",
                   HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        while True:
            try:
                name, fold, seed = pending.get_nowait()
            except queue.Empty:
                return
            label = f"{name}_fold{fold}_seed{seed}"
            start = time.time()
            log_path = ROOT / "runs" / "foundation" / "logs" / f"{label}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            command = [python, str(ROOT / "scripts" / "run_foundation.py"), "one",
                       "--foundation", name, "--fold", str(fold), "--seed", str(seed)]
            with log_path.open("w") as log:
                code = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env).returncode
            with lock:
                (failed if code else done).append(label)
                status = "ok" if code == 0 else f"FAILED {code}"
                print(f"[{time.strftime('%H:%M:%S')}] gpu{gpu} {label} {status} "
                      f"{(time.time() - start) / 60:.1f} min; {len(done)} ok, {len(failed)} failed, "
                      f"{pending.qsize()} queued", flush=True)

    threads = [threading.Thread(target=worker, args=(gpu,)) for gpu in gpus]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if failed:
        print("Failed:", " ".join(failed), flush=True)
        sys.exit(1)


def summarize():
    print("foundation mean of 5-fold means, sd across seeds")
    for name in ("ctranspath", "uni"):
        print(name)
        for key in ("accuracy", "macro_f1", "macro_auroc", "nll", "ece_15_bins"):
            seed_means = []
            for seed in SEEDS:
                values = []
                for fold in FOLDS:
                    path = ROOT / "runs" / "foundation" / name / f"fold{fold}_seed{seed}" / "test" / "metrics.json"
                    values.append(json.loads(path.read_text())[key])
                seed_means.append(float(np.mean(values)))
            array = np.asarray(seed_means)
            print(f"  {key:16} {array.mean():.3f} ± {array.std(ddof=1):.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", nargs="?", choices=("all", "one", "summary"), default="all")
    parser.add_argument("--foundation", choices=tuple(WEIGHTS))
    parser.add_argument("--fold", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--gpus", default="0,1,2,3")
    args = parser.parse_args()
    if args.command == "one":
        run_one(args.foundation, args.fold, args.seed)
    elif args.command == "summary":
        summarize()
    else:
        jobs = pending_jobs()
        print(f"{len(jobs)} foundation runs", flush=True)
        if jobs:
            run_parallel(jobs, [int(gpu) for gpu in args.gpus.split(",") if gpu != ""])
        summarize()


if __name__ == "__main__":
    main()
