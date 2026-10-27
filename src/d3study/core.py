from __future__ import annotations

import contextlib
import hashlib
import json
import os
import platform
import re
import socket
import threading
import time
import uuid
from pathlib import Path

GENERATORS = ("generator_small", "generator_large")
SELECTORS = ("decision_small", "decision_large")


def active_generators(cfg):
    return _active_roles(cfg, "active_generators", GENERATORS)


def active_selectors(cfg):
    return _active_roles(cfg, "active_selectors", SELECTORS)


def _active_roles(cfg, key, allowed):
    chosen = cfg.get("runtime", {}).get(key)
    if not chosen:
        return allowed
    if not isinstance(chosen, list) or not chosen or any(name not in allowed for name in chosen):
        raise StudyError(f"{key} must list known roles")
    return tuple(chosen)
COMPLETE = "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"


class StudyError(RuntimeError):
    pass


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    try:
        with tmp.open("w", encoding="utf-8", newline="\n") as out:
            out.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
            out.flush()
            os.fsync(out.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def event(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("a", encoding="utf-8", newline="\n") as out:
        out.write(canonical({"timestamp_unix": time.time(), **data}) + "\n")
        out.flush()
        os.fsync(out.fileno())


def safe_id(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,180}", value) or value in (".", ".."):
        raise StudyError("Invalid artifact/task identifier")
    return value


def parse_action(content):
    if not isinstance(content, str) or content.count("```") != 2:
        return None
    match = re.fullmatch(r"\s*```bash[ \t]*\r?\n([\s\S]+?)\r?\n```[ \t\r\n]*", content)
    return match[1] if match and match[1].strip() else None


def seed_for(*parts):
    return int(hashlib.sha256(canonical(parts).encode()).hexdigest()[:8], 16) % (2**31)


def prepend_library_path(paths):
    """Put configured CUDA libraries ahead of the inherited linker path."""
    prefix = [path for path in paths if isinstance(path, str) and path]
    if not prefix:
        return
    inherited = os.environ.get("LD_LIBRARY_PATH")
    os.environ["LD_LIBRARY_PATH"] = os.pathsep.join([*prefix, *([inherited] if inherited else [])])


def clip(text, limit):
    if len(text) <= limit:
        return text
    half = limit // 2
    return text[:half] + "\n[output truncated; full log stored separately]\n" + text[-half:]


def load_config(path):
    path = Path(path).resolve()
    cfg = read_json(path)
    if cfg.get("selection_level") != "completed_trajectory_final_patch":
        raise StudyError("Only completed-trajectory selection is implemented")
    if cfg.get("train_or_finetune_models") is not False:
        raise StudyError("This runner does not train or fine-tune models")
    for role in (*GENERATORS, *SELECTORS):
        model = cfg.get("models", {}).get(role, {})
        if role in GENERATORS and model.get("backend") == "ollama":
            name = model.get("ollama_model")
            if not isinstance(name, str) or not name.strip() or any(c.isspace() for c in name):
                raise StudyError(f"{role} needs an Ollama model tag")
        elif not isinstance(model.get("repo"), str) or not re.fullmatch(r"[0-9a-f]{40}", model.get("revision", "")):
            raise StudyError(f"{role} needs a repo and full pinned revision")
        if role in GENERATORS and (model.get("thinking") is not True or model.get("chat_template_kwargs", {}).get("enable_thinking") is not True):
            raise StudyError(f"{role} must explicitly enable native thinking")
    runtime = cfg.get("runtime")
    if not isinstance(runtime, dict) or not isinstance(runtime.get("workspace"), str):
        raise StudyError("GPU runtime configuration missing; use configs/gpu.example.json")
    n = cfg.get("candidate_pool", {}).get("main_n")
    if type(n) is not int or n != 3:
        raise StudyError("This study requires exactly three complete trajectories per candidate pool")
    limits = cfg["rollout_limits"]
    for key in ("decision_steps", "command_timeout_seconds", "active_rollout_timeout_seconds", "model_request_timeout_seconds", "max_new_tokens_per_action", "generator_max_context_tokens", "max_total_generated_tokens_per_rollout"):
        if isinstance(limits.get(key), bool) or not isinstance(limits.get(key), int) or limits[key] <= 0:
            raise StudyError(f"Invalid positive limit: {key}")
    if limits["max_new_tokens_per_action"] > limits["generator_max_context_tokens"]:
        raise StudyError("A reply cannot request more tokens than the model context")
    for role in (*GENERATORS, *SELECTORS):
        if role not in runtime.get("gpu_roles", {}):
            raise StudyError(f"GPU role mapping missing: {role}")
        gpu = runtime["gpu_roles"][role]
        if not isinstance(gpu.get("visible_devices"), str) or not gpu["visible_devices"].strip():
            raise StudyError(f"Explicit GPU IDs are required for {role}")
        if role in GENERATORS:
            if gpu.get("max_model_len") != limits["generator_max_context_tokens"]:
                raise StudyError("Generator service context must equal the frozen rollout context")
            model = cfg["models"][role]
            if model.get("backend") == "ollama":
                service = runtime.get("generator_services", {}).get(role, {})
                if service.get("num_ctx") != limits["generator_max_context_tokens"]:
                    raise StudyError("Ollama num_ctx must equal the generator context budget")
                ollama = runtime.get("ollama")
                if not isinstance(ollama, dict) or not isinstance(ollama.get("executable"), str) or not isinstance(ollama.get("models_dir"), str):
                    raise StudyError("Ollama runtime needs an executable and models_dir")
            elif gpu.get("tensor_parallel_size", 0) < 1:
                raise StudyError("Generator tensor parallel size must be positive")
        elif gpu.get("question_context_tokens") != cfg["decision_input"]["common_question_context_budget_tokens"]:
            raise StudyError("Both selectors must use the common decision context budget")
    workspace = Path(runtime["workspace"]).expanduser()
    workspace = workspace if workspace.is_absolute() else path.parent / workspace
    return cfg, workspace.resolve()


def managed_service(cfg, workspace, role):
    """Associate the endpoint with our live local pinned launch, not just its alias."""
    directory = workspace / "services" / role
    try:
        deployment = read_json(directory / "deployment.json")
        state = read_json(directory / "status.json")
        boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        if any(deployment[key] != expected for key, expected in (
            ("model", cfg["models"][role]), ("gpu", cfg["runtime"]["gpu_roles"][role]),
            ("service", cfg["runtime"]["generator_services"][role]),
            ("hostname", socket.gethostname()), ("boot_id", boot))):
            raise ValueError()
        if state.get("state") != "running" or state.get("pid") != deployment["controller_pid"] or state.get("boot_id") != boot or time.time()-state["heartbeat_unix"] >= 30:
            raise ValueError()
        model = cfg["models"][role]
        if model.get("backend") == "ollama":
            if deployment.get("ollama_model") != model.get("ollama_model"):
                raise ValueError()
        elif Path(deployment["snapshot"]).name != model["revision"]:
            raise ValueError()
        os.kill(deployment["server_pid"], 0)
        os.kill(deployment["controller_pid"], 0)
    except (OSError, KeyError, TypeError, ValueError):
        raise StudyError("Start this generator with serve-generator on the target first; live pinned deployment could not be verified") from None
    return {"deployment": deployment, "deployment_sha256": digest(deployment), "verification": "trusted local managed process plus endpoint model identity; not remote attestation"}


def require_execution(cfg):
    if cfg.get("execution_enabled") is not True:
        raise StudyError("Execution is disabled. On the target device, copy the GPU config and explicitly set execution_enabled=true.")


def require_linux():
    if platform.system() != "Linux" or platform.machine().lower() not in ("x86_64", "amd64"):
        raise StudyError("Model serving and SWE-bench execution require the documented Linux x86_64 target")


@contextlib.contextmanager
def run_lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open("a+b")
    try:
        try:
            if os.name == "nt":
                import msvcrt
                if path.stat().st_size == 0:
                    stream.write(b"0")
                    stream.flush()
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise StudyError("Another process holds this stage lock") from None
        yield
    finally:
        stream.close()  # OS locks are released even after crashes/reboots.


def code_hashes():
    root = Path(__file__).parent
    return {str(p.relative_to(root)): file_digest(p) for p in sorted(root.rglob("*")) if p.is_file() and p.suffix in (".py", ".txt")}


def data_spec(cfg, stage):
    runtime = cfg["runtime"]
    return {"stage": stage, "dataset": cfg["dataset"], "debug_tasks": runtime.get("debug_tasks", 2), "debug_dataset": runtime.get("debug_dataset", "princeton-nlp/SWE-bench"), "debug_dataset_revision": runtime.get("debug_dataset_revision"), "debug_split": runtime.get("debug_split", "dev")}


def freeze_run(cfg, workspace, stage):
    run = workspace / "runs" / safe_id(stage)
    run.mkdir(parents=True, exist_ok=True)
    data = workspace / "data" / stage
    if not (data / "manifest.json").exists():
        raise StudyError("Prepare the task manifest first")
    data_manifest = read_json(data / "manifest.json")
    if data_manifest.get("preparation_spec_sha256") != digest(data_spec(cfg, stage)):
        raise StudyError("Prepared data does not match the current sampling configuration; use a new workspace")
    for name, expected in data_manifest["hashes"].items():
        if file_digest(data / name) != expected:
            raise StudyError("Prepared task data changed")
    if not (data / "images.json").exists():
        raise StudyError("Prepare and freeze Docker images first")
    manifest = {"config": cfg, "config_sha256": digest(cfg), "source_hashes": code_hashes(), "data_manifest_sha256": file_digest(data / "manifest.json"), "image_manifest_sha256": file_digest(data / "images.json")}
    path = run / "frozen.json"
    if path.exists():
        if read_json(path) != manifest:
            raise StudyError("Frozen configuration, source or data changed; use a new workspace for a revised experiment")
    else:
        atomic_json(path, manifest)
    return run


class Heartbeat:
    def __init__(self, run, phase, **details):
        self.path = Path(run) / "status.json"
        boot = Path("/proc/sys/kernel/random/boot_id")
        self.data = {"phase": phase, "state": "running", "pid": os.getpid(), "hostname": socket.gethostname(), "boot_id": boot.read_text().strip() if boot.exists() else None, **details}
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def _write(self):
        atomic_json(self.path, {**self.data, "heartbeat_unix": time.time()})

    def _loop(self):
        while not self.stop.wait(10):
            self._write()

    def __enter__(self):
        self._write()
        self.thread.start()
        return self

    def __exit__(self, typ, value, tb):
        self.stop.set()
        self.thread.join()
        self.data["state"] = "finished" if typ is None else "interrupted" if issubclass(typ, KeyboardInterrupt) else "failed"
        self._write()
