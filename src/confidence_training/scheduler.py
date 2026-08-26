"""Deadline-aware longest-processing-time scheduler for sweep jobs."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Sequence

from .artifacts import atomic_write_json


TERMINAL_JOB_STATES = {"complete", "failed", "deferred", "deadline_stopped"}


def validate_jobs(jobs: Sequence[dict[str, Any]]) -> None:
    identifiers: set[str] = set()
    for job in jobs:
        job_id = str(job.get("job_id", ""))
        if not job_id or job_id in identifiers:
            raise ValueError(f"Job IDs must be nonempty and unique: {job_id!r}")
        identifiers.add(job_id)
        command = job.get("command")
        if not isinstance(command, list) or not command or not all(
            isinstance(token, str) and token for token in command
        ):
            raise ValueError(f"Job {job_id} command must be a nonempty argv list")
        if command[:2] != ["uv", "run"]:
            raise ValueError(f"Job {job_id} must execute through 'uv run'")
        predicted = float(job.get("predicted_seconds", 0))
        if predicted <= 0:
            raise ValueError(f"Job {job_id} predicted_seconds must be positive")


def lpt_assignment(
    jobs: Sequence[dict[str, Any]], max_workers: int = 3
) -> list[dict[str, Any]]:
    if max_workers <= 0:
        raise ValueError("max_workers must be positive")
    validate_jobs(jobs)
    lanes: list[dict[str, Any]] = [
        {"worker": index, "predicted_seconds": 0.0, "job_ids": []}
        for index in range(max_workers)
    ]
    ordered = sorted(
        jobs, key=lambda job: (-float(job["predicted_seconds"]), str(job["job_id"]))
    )
    for job in ordered:
        lane = min(lanes, key=lambda value: (value["predicted_seconds"], value["worker"]))
        lane["job_ids"].append(str(job["job_id"]))
        lane["predicted_seconds"] += float(job["predicted_seconds"])
    return lanes


def load_plan(path: str | Path) -> dict[str, Any]:
    plan = json.loads(Path(path).read_text())
    if int(plan.get("plan_version", 0)) != 1:
        raise ValueError("Unsupported or missing plan_version")
    jobs = plan.get("jobs")
    if not isinstance(jobs, list):
        raise ValueError("Sweep plan must contain a jobs list")
    validate_jobs(jobs)
    return plan


def _initial_state(plan: dict[str, Any]) -> dict[str, Any]:
    return {
        "scheduler_state_version": 1,
        "jobs": {
            str(job["job_id"]): {
                "status": "pending",
                "predicted_seconds": float(job["predicted_seconds"]),
                "command": job["command"],
            }
            for job in plan["jobs"]
        },
    }


def run_plan(
    plan: dict[str, Any],
    *,
    state_path: str | Path,
    log_dir: str | Path,
    max_workers: int = 3,
    deadline_seconds: float,
    reserve_seconds: float = 300.0,
    popen_factory: Callable[..., Any] = subprocess.Popen,
    monotonic: Callable[[], float] = time.monotonic,
    poll_interval: float = 0.1,
) -> dict[str, Any]:
    """Run globally queued LPT jobs without launching beyond the hard deadline."""

    if max_workers <= 0 or deadline_seconds <= 0 or reserve_seconds < 0:
        raise ValueError("Invalid worker/deadline/reserve settings")
    validate_jobs(plan["jobs"])
    state_file = Path(state_path)
    if state_file.exists():
        state = json.loads(state_file.read_text())
        for job_state in state["jobs"].values():
            if job_state["status"] == "running":
                job_state["status"] = "pending"
    else:
        state = _initial_state(plan)
    log_root = Path(log_dir)
    log_root.mkdir(parents=True, exist_ok=True)
    job_lookup = {str(job["job_id"]): job for job in plan["jobs"]}
    pending = sorted(
        [
            job_lookup[job_id]
            for job_id, value in state["jobs"].items()
            if value["status"] == "pending"
        ],
        key=lambda job: (-float(job["predicted_seconds"]), str(job["job_id"])),
    )
    active: dict[str, tuple[Any, Any]] = {}
    started = monotonic()
    hard_deadline = started + deadline_seconds
    state["started_monotonic"] = started
    state["deadline_seconds"] = deadline_seconds
    state["reserve_seconds"] = reserve_seconds

    while pending or active:
        now = monotonic()
        remaining = hard_deadline - now
        if remaining <= 0:
            for job_id, (process, handle) in active.items():
                process.terminate()
                process.wait()
                handle.close()
                state["jobs"][job_id]["status"] = "deadline_stopped"
                state["jobs"][job_id]["returncode"] = process.returncode
            active.clear()
            for job in pending:
                state["jobs"][str(job["job_id"])]["status"] = "deferred"
            pending.clear()
            break

        launched = False
        while len(active) < max_workers and pending:
            launch_index = next(
                (
                    index
                    for index, job in enumerate(pending)
                    if float(job["predicted_seconds"]) <= remaining - reserve_seconds
                ),
                None,
            )
            if launch_index is None:
                for job in pending:
                    state["jobs"][str(job["job_id"])]["status"] = "deferred"
                pending.clear()
                break
            job = pending.pop(launch_index)
            job_id = str(job["job_id"])
            log_path = log_root / f"{job_id}.log"
            handle = log_path.open("ab")
            process = popen_factory(job["command"], stdout=handle, stderr=subprocess.STDOUT)
            active[job_id] = (process, handle)
            state["jobs"][job_id].update(
                {"status": "running", "started_monotonic": now, "log_path": str(log_path)}
            )
            launched = True

        finished: list[str] = []
        for job_id, (process, handle) in active.items():
            returncode = process.poll()
            if returncode is None:
                continue
            handle.close()
            state["jobs"][job_id].update(
                {
                    "status": "complete" if returncode == 0 else "failed",
                    "returncode": returncode,
                    "finished_monotonic": monotonic(),
                }
            )
            finished.append(job_id)
        for job_id in finished:
            del active[job_id]
        state["updated_monotonic"] = monotonic()
        atomic_write_json(state_file, state)
        if active and not finished and not launched:
            time.sleep(poll_interval)

    state["finished_monotonic"] = monotonic()
    atomic_write_json(state_file, state)
    return state
