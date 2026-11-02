"""Write a fixed experiment plan; only --execute starts training and test evaluation."""

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

from castgate.engine import load_config

VARIANTS = {
    "cnn": {"mode": "cnn"},
    "transformer": {"mode": "transformer"},
    "mean": {"mode": "mean"},
    "learned_gate": {"mode": "learned_gate"},
    "reliability_joint": {"mode": "reliability_gate", "reliability": "joint"},
    "reliability_accuracy": {"mode": "reliability_gate", "reliability": "accuracy"},
    "reliability_stability": {"mode": "reliability_gate", "reliability": "stability"},
    "mean_no_stain": {"mode": "mean", "stain_strength": 0.0},
}


def run_parallel(runs, gpus, jobs_per_gpu, log_dir):
    """Each worker owns one GPU slot; a failed run is logged and does not stop the others."""
    log_dir.mkdir(parents=True, exist_ok=True)
    pending = queue.Queue()
    for run in runs:
        pending.put(run)
    failed, done, lock = [], [], threading.Lock()

    def worker(gpu):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
        # cuBLAS is only deterministic with a fixed workspace size.
        env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        while True:
            try:
                name, commands = pending.get_nowait()
            except queue.Empty:
                return
            start = time.time()
            with (log_dir / f"{name}.log").open("w") as log:
                code = 0
                for command in commands:
                    code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env).returncode
                    if code:
                        break
            with lock:
                (failed if code else done).append(name)
                status = f"FAILED (exit {code})" if code else "ok"
                print(f"[{time.strftime('%H:%M:%S')}] gpu{gpu} {name} {status} "
                      f"{(time.time() - start) / 60:.1f} min; {len(done)} ok, {len(failed)} failed, "
                      f"{len(runs) - len(done) - len(failed)} remaining", flush=True)

    threads = [threading.Thread(target=worker, args=(gpu,)) for gpu in gpus for _ in range(jobs_per_gpu)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if failed:
        print(f"Failed runs: {sorted(failed)}", flush=True)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/lunghist700.json")
    parser.add_argument("--output", default="runs/experiments")
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--only", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--gpus", nargs="+", type=int,
                        help="Run in parallel, one worker per GPU slot; default is sequential")
    parser.add_argument("--jobs-per-gpu", type=int, default=1)
    args = parser.parse_args()
    base = load_config(args.config)
    out = Path(args.output).resolve()
    plan_dir = out / "plan"
    if plan_dir.exists():
        raise FileExistsError(f"Plan already exists: {plan_dir}; choose a new output directory")
    plan_dir.mkdir(parents=True)
    commands, runs = [], []
    for variant in args.only:
        for seed in args.seeds:
            for fold in args.folds:
                run_name = f"{variant}_fold{fold}_seed{seed}"
                config = dict(base, **VARIANTS[variant])
                config.update(fold=fold, seed=seed, output=str(out / variant / f"fold{fold}_seed{seed}"))
                file = plan_dir / f"{run_name}.json"
                file.write_text(json.dumps(config, indent=2))
                train = [sys.executable, "-m", "castgate", "train", "--config", str(file)]
                evaluate = [sys.executable, "-m", "castgate", "evaluate", "--checkpoint",
                            str(Path(config["output"]) / "best.pt"), "--output",
                            str(Path(config["output"]) / "test")]
                commands += [train, evaluate]
                runs.append(((seed, fold, variant), run_name, [train, evaluate]))
    (plan_dir / "commands.json").write_text(json.dumps(commands, indent=2))
    print(f"Plan: {len(commands)//2} runs, {plan_dir}", flush=True)
    if args.execute and args.gpus:
        # Seed-major order so every variant of a seed finishes before the next seed starts.
        run_parallel([(name, cmds) for _, name, cmds in sorted(runs)], args.gpus, args.jobs_per_gpu,
                     out / "logs")
    elif args.execute:
        for command in commands:
            subprocess.run(command, check=True)
    else:
        print("No training started. Review configs and regenerate under a new --output with --execute.")


if __name__ == "__main__":
    main()
