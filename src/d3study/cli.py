from __future__ import annotations
import argparse
import json
import os
import signal
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit

from .backends import BackendError, build_vllm_command
from .core import (GENERATORS, SELECTORS, Heartbeat, StudyError, atomic_json, canonical,
                   digest, load_config, read_json, require_execution, require_linux, run_lock)


def serve(cfg, workspace, role, print_command=False):
    model = cfg["models"][role]
    gpu = cfg["runtime"]["gpu_roles"][role]
    service = cfg["runtime"]["generator_services"][role]
    parsed = urlsplit(service["base_url"])
    if parsed.hostname not in ("127.0.0.1", "localhost"):
        raise StudyError("This runner requires a managed loopback generator service")
    if print_command:
        snapshot = "/PINNED_HF_CACHE/snapshots/" + model["revision"]
    else:
        require_execution(cfg)
        require_linux()
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu["visible_devices"]
        from .device import cuda_inventory
        inventory = cuda_inventory(strict=True)
        if len(inventory["devices"]) < gpu.get("tensor_parallel_size", 1):
            raise StudyError("Not enough visible GPUs for configured tensor parallel size")
        from huggingface_hub import snapshot_download
        snapshot = snapshot_download(repo_id=model["repo"], revision=model["revision"], cache_dir=cfg["runtime"].get("hf_cache_dir"))
        if Path(snapshot).resolve().name != model["revision"]:
            raise StudyError("Model snapshot did not resolve to the pinned SHA")
    command = build_vllm_command(model, gpu, str(snapshot)) + ["--host", "127.0.0.1", "--port", str(parsed.port or 8000)]
    if print_command:
        return {"command": command, "CUDA_VISIBLE_DEVICES": gpu["visible_devices"], "executed": False, "snapshot_path_note": "placeholder only; real launch resolves the pinned HF snapshot"}
    executable = shutil.which("vllm", path=str(Path(sys.executable).parent))
    if executable is None:
        raise StudyError("vLLM executable missing from this Python environment; use the generator environment")
    command[0] = executable
    # Avoid a server accidentally taking the other generator's port.
    with run_lock(workspace / "services" / f"{role}.lock"):
        directory = workspace / "services" / role
        directory.mkdir(parents=True, exist_ok=True)
        with Heartbeat(directory, "serve", generator=role) as heartbeat, (directory / "server.log").open("a", encoding="utf-8") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            atomic_json(directory / "deployment.json", {"model": model, "gpu": gpu, "service": service, "snapshot": str(snapshot), "command": command, "server_pid": process.pid, "controller_pid": os.getpid(), "hostname": heartbeat.data["hostname"], "boot_id": heartbeat.data["boot_id"], "environment_credentials": "not recorded", "target_inventory": inventory})
            try:
                code = process.wait()
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
            atomic_json(directory / "exit.json", {"exit_code": code})
            if code:
                raise StudyError("vLLM server exited with an error; inspect its server.log")
    return {"service": role, "exit_code": code}


def status(cfg, workspace, stage):
    run = workspace / "runs" / stage
    state = read_json(run / "status.json") if (run / "status.json").exists() else {"state": "not_started"}
    if "heartbeat_unix" in state:
        state["heartbeat_age_seconds"] = max(0, time.time()-state["heartbeat_unix"])
        boot = Path("/proc/sys/kernel/random/boot_id")
        same_host = state.get("hostname") == socket.gethostname()
        same_boot = state.get("boot_id") is None or (boot.exists() and boot.read_text().strip() == state["boot_id"])
        alive = None
        if same_host and same_boot:
            try:
                os.kill(state["pid"], 0)
                alive = True
            except (OSError, ProcessLookupError):
                alive = False
        elif same_host:
            alive = False
        state["recorded_process_alive_on_this_boot"] = alive
        state["currently_running_confirmed"] = state.get("state") == "running" and alive is True and state["heartbeat_age_seconds"] < 30
    state["complete_candidate_slots"] = len(list(run.glob("pools/*/*/candidate_*/result.json")))
    state["decision_records"] = len(list((run / "decisions").rglob("*.json"))) if (run / "decisions").exists() else 0
    state["report"] = str(run / "analysis" / "report.md") if (run / "analysis" / "report.md").exists() else None
    return state


def parser():
    p = argparse.ArgumentParser(description="Bidirectional GPU trajectory Best-of-3 empirical study")
    p.add_argument("--config", type=Path, required=True)
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("validate-config", help="Validate portable configuration without importing GPU libraries")
    doc = sub.add_parser("doctor", help="Read-only target inventory; no model loading/inference")
    doc.add_argument("--output", type=Path, required=True)
    serve_p = sub.add_parser("serve-generator", help="Foreground pinned vLLM GPU server; use a target job manager")
    serve_p.add_argument("--generator", choices=GENERATORS, required=True)
    serve_p.add_argument("--print-command", action="store_true", help="Print only, without downloading or running anything")
    for cmd in ("prepare-data", "prepare-images", "generate", "select", "evaluate", "analyze", "status"):
        c = sub.add_parser(cmd)
        c.add_argument("--stage", choices=("debug", "pilot"), required=True)
        if cmd == "prepare-data":
            c.add_argument("--input-json", type=Path)
            c.add_argument("--verified-json", type=Path)
        if cmd == "generate":
            c.add_argument("--generator", choices=GENERATORS, required=True)
        if cmd == "select":
            c.add_argument("--selector", choices=SELECTORS, required=True)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        cfg, workspace = load_config(args.config)
        if args.command == "validate-config":
            result = {"valid": True, "execution_enabled": cfg.get("execution_enabled"), "workspace": str(workspace), "config_sha256": digest(cfg)}
        elif args.command == "doctor":
            from .device import doctor
            result = doctor(cfg, workspace, args.output)
        elif args.command == "serve-generator":
            result = serve(cfg, workspace, args.generator, args.print_command)
        elif args.command == "status":
            result = status(cfg, workspace, args.stage)
        elif args.command in ("evaluate", "analyze"):
            from .reports import evaluate, analyze
            result = (evaluate if args.command == "evaluate" else analyze)(cfg, workspace, args.stage)
        else:
            from .workflow import prepare_data, prepare_images, generate, select
            if args.command == "prepare-data":
                result = prepare_data(cfg, workspace, args.stage, args.input_json, args.verified_json)
            elif args.command == "prepare-images":
                result = prepare_images(cfg, workspace, args.stage)
            elif args.command == "generate":
                result = generate(cfg, workspace, args.stage, args.generator)
            else:
                result = select(cfg, workspace, args.stage, args.selector)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (StudyError, BackendError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
