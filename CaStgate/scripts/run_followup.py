"""Train the pre-specified follow-up runs.

Checkpoints are removed after a successful test evaluation. Metrics and
predictions are kept. Disk headroom on this machine is only a few gigabytes.
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

ROOT = Path(__file__).resolve().parents[1]
JOINT = {"mode": "reliability_gate", "reliability": "joint"}


def variants():
    found = {
        "shuffle_joint": dict(JOINT, shuffle_gate_target=True),
        "beta_0.5": dict(JOINT, stability_beta=0.5),
        "beta_2": dict(JOINT, stability_beta=2.0),
        "tau_0.25": dict(JOINT, temperature=0.25),
        "tau_1": dict(JOINT, temperature=1.0),
        "lambda_0": dict(JOINT, lambda_gate=0.0),
        "lambda_0.25": dict(JOINT, lambda_gate=0.25),
        "lambda_1": dict(JOINT, lambda_gate=1.0),
    }
    for direction, train_mag, test_mag in (("20to40", "20x", "40x"), ("40to20", "40x", "20x")):
        for name, extra in (("cnn", {"mode": "cnn"}), ("transformer", {"mode": "transformer"}),
                            ("mean", {"mode": "mean"}), ("joint", dict(JOINT))):
            found[f"xmag_{direction}_{name}"] = dict(extra, train_magnification=train_mag,
                                                     test_magnification=test_mag)
    return found


def run_parallel(jobs, gpus):
    pending = queue.Queue()
    for job in jobs:
        pending.put(job)
    failed, done, lock = [], [], threading.Lock()

    def worker(gpu):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), CUBLAS_WORKSPACE_CONFIG=":4096:8",
                   HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        while True:
            try:
                name, commands, checkpoint = pending.get_nowait()
            except queue.Empty:
                return
            start = time.time()
            log_path = ROOT / "runs" / "followup" / "logs" / f"{name}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("w") as log:
                code = 0
                for command in commands:
                    code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env).returncode
                    if code:
                        break
            if code == 0 and checkpoint.exists():
                checkpoint.unlink()
            with lock:
                (failed if code else done).append(name)
                status = "ok" if code == 0 else f"FAILED {code}"
                print(f"[{time.strftime('%H:%M:%S')}] gpu{gpu} {name} {status} "
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", nargs="+", type=int, required=True)
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--only", nargs="+", default=None)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    selected = variants()
    names = args.only or list(selected)
    unknown = set(names) - set(selected)
    if unknown:
        raise SystemExit(f"Unknown variants: {sorted(unknown)}")
    # Match the completed 120-run protocol, including local backbone weights.
    base = json.loads((ROOT / "runs" / "experiments" / "reliability_joint" / "fold0_seed42" / "config.json").read_text())
    for key in ("output", "fold", "seed"):
        base.pop(key, None)
    out = ROOT / "runs" / "followup"
    jobs = []
    for name in names:
        for seed in args.seeds:
            for fold in args.folds:
                run = out / name / f"fold{fold}_seed{seed}"
                if (run / "test" / "metrics.json").exists():
                    continue
                if (run / "config.json").exists() and not (run / "best.pt").exists():
                    shutil.rmtree(run)
                config = dict(base, **selected[name])
                config.update(fold=fold, seed=seed, output=str(run))
                config_path = out / "plan" / f"{name}_fold{fold}_seed{seed}.json"
                config_path.parent.mkdir(parents=True, exist_ok=True)
                config_path.write_text(json.dumps(config, indent=2))
                train = [sys.executable, "-m", "castgate", "train", "--config", str(config_path)]
                evaluate = [sys.executable, "-m", "castgate", "evaluate", "--checkpoint",
                            str(run / "best.pt"), "--output", str(run / "test")]
                commands = [] if (run / "best.pt").exists() else [train]
                commands.append(evaluate)
                jobs.append((f"{name}_fold{fold}_seed{seed}", commands, run / "best.pt"))
    print(f"Pending runs: {len(jobs)}", flush=True)
    if args.execute:
        run_parallel(jobs, args.gpus)


if __name__ == "__main__":
    main()
