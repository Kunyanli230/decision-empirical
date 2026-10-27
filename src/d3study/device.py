"""Read-only target inventory. No model loading, inference or CUDA allocations."""
from __future__ import annotations
import importlib
import importlib.metadata
import os
import platform
import shutil
import socket
import subprocess
import sys
from pathlib import Path
from .core import StudyError, atomic_json


def command_output(argv):
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=20)
        return {"exit_code": p.returncode, "stdout": p.stdout[-12000:], "stderr": p.stderr[-1000:]}
    except (OSError, subprocess.TimeoutExpired):
        return {"available": False}


def cuda_inventory(strict=False):
    try:
        torch = importlib.import_module("torch")
    except ImportError:
        if strict:
            raise StudyError("PyTorch is not installed in this target model environment") from None
        return {"available": False, "reason": "torch_not_installed"}
    result = {"torch_version": torch.__version__, "cuda_build": torch.version.cuda, "rocm_build": getattr(torch.version, "hip", None), "available": torch.cuda.is_available(), "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "devices": []}
    if result["available"]:
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            result["devices"].append({"index": i, "name": p.name, "total_memory_bytes": p.total_memory, "compute_capability": [p.major, p.minor]})
        result["bfloat16_supported"] = torch.cuda.is_bf16_supported()
    if strict and (not result["available"] or not result.get("bfloat16_supported")):
        raise StudyError("A visible BF16-capable GPU is required; CPU fallback is disabled")
    return result


def doctor(cfg, workspace, output):
    existing = workspace
    while not existing.exists():
        existing = existing.parent
    usage = shutil.disk_usage(existing)
    names = ("d3-bidirectional-study", "torch", "torchvision", "transformers", "vllm", "huggingface-hub", "datasets", "swebench", "requests", "flash-linear-attention")
    packages = {}
    for name in names:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    docker = cfg["runtime"]["docker"].get("executable", "docker")
    result = {"hostname": socket.gethostname(), "system": platform.platform(), "architecture": platform.machine(), "python": sys.version, "python_executable": sys.executable, "libc": platform.libc_ver(), "logical_cpu_count": os.cpu_count(),
              "disk": {"checked_path": str(existing), "total_bytes": usage.total, "free_bytes": usage.free}, "packages": packages,
              "torch_gpu": cuda_inventory(), "nvidia_smi": command_output(["nvidia-smi", "--query-gpu=index,name,uuid,memory.total,memory.free,driver_version", "--format=csv"]), "gpu_topology": command_output(["nvidia-smi", "topo", "-m"]),
              "docker_server": command_output([docker, "version", "--format", "{{.Server.Version}}"]), "docker_storage": command_output([docker, "info", "--format", "{{.Architecture}} {{.DockerRootDir}}"]),
              "workspace": str(workspace), "runtime": cfg["runtime"], "credential_presence_only": {key: bool(os.environ.get(key)) for key in ("HF_TOKEN", "HTTP_PROXY", "HTTPS_PROXY")},
              "note": "Inventory only; does not prove model fit, thinking behavior, GPU inference compatibility or benchmark correctness."}
    if hasattr(os, "sysconf"):
        try:
            result["ram_bytes"] = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        except (OSError, ValueError):
            pass
    atomic_json(output, result)
    return result
