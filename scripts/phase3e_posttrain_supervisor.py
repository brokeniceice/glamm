#!/usr/bin/env python3
"""Resumable unattended Phase 3E post-training evaluation supervisor.

Runs only the preregistered internal-validation workflow.  It never reads the
sealed internal-test or official1000 populations.
"""

from __future__ import annotations

import fcntl
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/phase3e_joint_language_mask_posttraining.yaml"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


class Supervisor:
    def __init__(self) -> None:
        cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
        self.root = (ROOT / cfg["experiment"]["output_root"]).resolve()
        self.runtime = self.root / "posttrain_supervisor"
        self.state_path = self.runtime / "state.json"
        self.logs = self.runtime / "logs"
        self.logs.mkdir(parents=True, exist_ok=True)
        self.state = {
            "status": "RUNNING",
            "started_utc": now(),
            "updated_utc": now(),
            "pid": os.getpid(),
            "canonical_prompt_required": {"detection": True, "G0": True, "TF": True},
            "generation_batch_size": 1,
            "internal_test_used": False,
            "official1000_used": False,
            "stages": [],
        }
        dump(self.state_path, self.state)

    def record(self, name: str, status: str, **extra) -> None:
        self.state["updated_utc"] = now()
        self.state["stages"].append({"name": name, "status": status, "utc": now(), **extra})
        dump(self.state_path, self.state)

    def command(self, name: str, arguments: list[str], *, gpu: int | None = None) -> None:
        self.record(name, "STARTED", gpu=gpu, command=arguments)
        env = os.environ.copy()
        if gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        log_path = self.logs / f"{name}.log"
        with log_path.open("a", encoding="utf-8") as log:
            completed = subprocess.run(
                [sys.executable, *arguments], cwd=ROOT, env=env,
                stdout=log, stderr=subprocess.STDOUT,
            )
        if completed.returncode:
            self.record(name, "FAILED", returncode=completed.returncode, log=str(log_path))
            raise RuntimeError(f"{name} failed with return code {completed.returncode}; see {log_path}")
        self.record(name, "COMPLETE", returncode=0, log=str(log_path))

    def launch(self, name: str, arguments: list[str], *, gpu: int) -> tuple[subprocess.Popen, object, Path]:
        self.record(name, "STARTED", gpu=gpu, command=arguments)
        env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        log_path = self.logs / f"{name}.log"
        log = log_path.open("a", encoding="utf-8")
        process = subprocess.Popen(
            [sys.executable, *arguments], cwd=ROOT, env=env,
            stdout=log, stderr=subprocess.STDOUT,
        )
        return process, log, log_path

    def finish_process(self, name: str, process: subprocess.Popen, log, log_path: Path) -> None:
        returncode = process.wait(); log.close()
        if returncode:
            self.record(name, "FAILED", returncode=returncode, log=str(log_path))
            raise RuntimeError(f"{name} failed with return code {returncode}; see {log_path}")
        self.record(name, "COMPLETE", returncode=0, log=str(log_path))

    def selectors(self) -> None:
        selector_root = self.root / "evaluation/selector"
        sft_done = selector_root / "SFT_CONT_selector.json"
        joint_done = selector_root / "JOINT_selector.json"
        processes: dict[str, tuple[subprocess.Popen, object, Path]] = {}
        if not sft_done.is_file():
            processes["selector_sft_cont"] = self.launch(
                "selector_sft_cont",
                ["scripts/phase3e_validate_select.py", "--arm", "SFT_CONT", "--physical-gpu", "1"],
                gpu=1,
            )
        p1_metrics = selector_root / "P1_FROZEN/selection_metrics.json"
        while not p1_metrics.is_file() and "selector_sft_cont" in processes:
            process = processes["selector_sft_cont"][0]
            if process.poll() is not None:
                self.finish_process("selector_sft_cont", *processes.pop("selector_sft_cont"))
                break
            time.sleep(10)
        if not p1_metrics.is_file() and not sft_done.is_file():
            raise RuntimeError("shared P1 canonical selector baseline was not completed")
        if not joint_done.is_file():
            processes["selector_joint"] = self.launch(
                "selector_joint",
                ["scripts/phase3e_validate_select.py", "--arm", "JOINT", "--physical-gpu", "2"],
                gpu=2,
            )
        for name, values in list(processes.items()):
            self.finish_process(name, *values)
        if not sft_done.is_file() or not joint_done.is_file():
            raise RuntimeError("selector completion artifacts are missing")

    def selected_tf(self) -> None:
        final_root = self.root / "evaluation/final"
        if not (final_root / "P1_FROZEN/summary.json").is_file():
            self.command(
                "tf_p1_frozen",
                ["scripts/phase3e_selected_tf_evaluate.py", "--arm", "P1_FROZEN", "--physical-gpu", "1"],
                gpu=1,
            )
        processes: dict[str, tuple[subprocess.Popen, object, Path]] = {}
        for name, arm, gpu in (("tf_sft_cont", "SFT_CONT", 1), ("tf_joint", "JOINT", 2)):
            if not (final_root / arm / "summary.json").is_file():
                processes[name] = self.launch(
                    name,
                    ["scripts/phase3e_selected_tf_evaluate.py", "--arm", arm, "--physical-gpu", str(gpu)],
                    gpu=gpu,
                )
        for name, values in list(processes.items()):
            self.finish_process(name, *values)

    def run(self) -> None:
        audit = self.root / "audits/training_fairness_runtime_audit.json"
        if not audit.is_file() or json.loads(audit.read_text(encoding="utf-8")).get("status") != "PASS":
            self.command("training_audit", ["scripts/phase3e_training_audit.py"])
        else:
            self.record("training_audit", "ALREADY_COMPLETE", artifact=str(audit))
        self.selectors()
        self.selected_tf()
        if not (self.root / "completion_manifest.json").is_file():
            self.command("finalize", ["scripts/phase3e_finalize.py"])
        completion = json.loads((self.root / "completion_manifest.json").read_text(encoding="utf-8"))
        if completion.get("status") != "COMPLETE":
            raise RuntimeError("completion manifest is not COMPLETE")
        self.state["status"] = "COMPLETE"
        self.state["completed_utc"] = now()
        self.state["updated_utc"] = now()
        self.state["primary_gate"] = completion.get("primary_gate")
        dump(self.state_path, self.state)


def main() -> None:
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    root = (ROOT / cfg["experiment"]["output_root"]).resolve()
    lock_path = root / "posttrain_supervisor/supervisor.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another Phase 3E post-training supervisor is already running") from error
        supervisor = Supervisor()
        try:
            supervisor.run()
        except BaseException as error:
            supervisor.state["status"] = "FAILED"
            supervisor.state["failed_utc"] = now()
            supervisor.state["updated_utc"] = now()
            supervisor.state["error"] = repr(error)
            supervisor.state["traceback"] = traceback.format_exc()
            dump(supervisor.state_path, supervisor.state)
            raise


if __name__ == "__main__":
    main()
