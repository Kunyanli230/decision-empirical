from __future__ import annotations
import copy
import hashlib
import json
import os
import random
import re
import time
from pathlib import Path, PurePosixPath

from .backends import BackendError, NativeD3Selector, OllamaGenerator, VLLMGenerator
from .core import (COMPLETE, GENERATORS, SELECTORS, Heartbeat, StudyError, active_generators, active_selectors, atomic_json,
                   canonical, clip, data_spec, digest, event, file_digest, freeze_run, parse_action,
                   managed_service, prepend_library_path, read_json, require_execution, require_linux, run_lock, safe_id, seed_for)
from .docker_env import DockerSession, EnrootSession, docker, enroot_import, file_sha256, inspect_image
from .step_dedup import should_verify


def _write_dataset(cfg, workspace, stage, rows, verified_ids, source):
    target = workspace / "data" / stage
    target.mkdir(parents=True, exist_ok=True)
    excluded = set(cfg["dataset"].get("excluded_previously_inspected_instances", []))
    if stage == "debug":
        excluded |= set(verified_ids)
    count = cfg["runtime"].get("debug_tasks", 2) if stage == "debug" else cfg["dataset"]["pilot_tasks"]
    eligible = sorted((r for r in rows if r["instance_id"] not in excluded), key=lambda r: r["instance_id"])
    if len({r["instance_id"] for r in eligible}) != len(eligible):
        raise StudyError("Duplicate task identifiers in source")
    random.Random(cfg["dataset"]["sampling_seed"]).shuffle(eligible)
    if len(eligible) < count:
        raise StudyError("Not enough disjoint tasks for the requested stage")
    chosen = eligible[:count]
    public = []
    for row in chosen:
        iid = safe_id(row["instance_id"])
        if not re.fullmatch(r"[0-9a-f]{40}", row["base_commit"]):
            raise StudyError("Task source contains an invalid base_commit")
        if not isinstance(row["problem_statement"], str) or not row["problem_statement"].strip():
            raise StudyError("Task has no issue statement")
        image = "docker.io/swebench/sweb.eval.x86_64." + iid.replace("__", "_1776_").lower() + ":latest"
        public.append({k: row[k] for k in ("instance_id", "repo", "base_commit", "problem_statement")})
        public[-1]["image_tag"] = image
    atomic_json(target / "public.json", public)
    atomic_json(target / "evaluation.json", chosen)
    if os.name != "nt":
        os.chmod(target / "evaluation.json", 0o600)
    manifest = {"stage": stage, "source": source, "preparation_spec_sha256": digest(data_spec(cfg, stage)), "selection_seed": cfg["dataset"]["sampling_seed"], "excluded_ids": sorted(excluded), "instance_ids": [r["instance_id"] for r in public], "count": count,
                "hashes": {name: file_digest(target / name) for name in ("public.json", "evaluation.json")}}
    atomic_json(target / "manifest.json", manifest)
    return manifest


def prepare_data(cfg, workspace, stage, input_json=None, verified_json=None):
    target = workspace / "data" / stage
    if (target / "manifest.json").exists():
        manifest = read_json(target / "manifest.json")
        if manifest.get("preparation_spec_sha256") != digest(data_spec(cfg, stage)):
            raise StudyError("Prepared data does not match current sampling settings")
        if input_json and manifest["source"].get("file_sha256") != file_digest(input_json):
            raise StudyError("Prepared offline source changed")
        if verified_json and manifest["source"].get("verified_file_sha256") != file_digest(verified_json):
            raise StudyError("Prepared offline Verified exclusion source changed")
        for name, expected in manifest["hashes"].items():
            if file_digest(target / name) != expected:
                raise StudyError("Previously frozen data was modified")
        return manifest
    if input_json:
        if stage == "debug" and not verified_json:
            raise StudyError("Offline debug preparation needs --verified-json to check disjointness")
        rows = read_json(input_json)
        verified = read_json(verified_json) if verified_json else []
        source = {"kind": "local_json", "file_sha256": file_digest(input_json), "verified_file_sha256": file_digest(verified_json) if verified_json else None}
    else:
        from datasets import load_dataset
        from huggingface_hub import HfApi
        api = HfApi()
        verified_repo = cfg["dataset"]["repo"]
        verified_revision = cfg["dataset"].get("revision") or api.dataset_info(verified_repo).sha
        verified = list(load_dataset(verified_repo, revision=verified_revision, split="test"))
        if stage == "debug":
            repo = cfg["runtime"].get("debug_dataset", "princeton-nlp/SWE-bench")
            revision = cfg["runtime"].get("debug_dataset_revision") or api.dataset_info(repo).sha
            split = cfg["runtime"].get("debug_split", "dev")
            rows = list(load_dataset(repo, revision=revision, split=split))
        else:
            repo, revision, split, rows = verified_repo, verified_revision, "test", verified
        source = {"kind": "huggingface", "repo": repo, "revision": revision, "split": split, "verified_repo": verified_repo, "verified_revision": verified_revision}
    return _write_dataset(cfg, workspace, stage, rows, [r["instance_id"] for r in verified], source)


def prepare_images(cfg, workspace, stage):
    require_execution(cfg)
    require_linux()
    target = workspace / "data" / stage
    rows = read_json(target / "public.json")
    path = target / "images.json"
    settings = cfg["runtime"]["docker"]
    if settings.get("runtime") == "enroot":
        return _prepare_enroot_images(cfg, target, rows, path, settings)
    exe = settings.get("executable", "docker")
    if path.exists():
        frozen = read_json(path)
        for info in frozen.values():
            if inspect_image(exe, info["tag"])["Id"] != info["id"]:
                raise StudyError("A frozen Docker tag changed")
        return frozen
    images = {}
    for row in rows:
        tag = row["image_tag"]
        docker(exe, ["pull", tag], timeout=cfg["runtime"].get("image_pull_timeout_seconds", 3600))
        meta = inspect_image(exe, tag)
        images[row["instance_id"]] = {"tag": tag, "id": meta["Id"], "repo_digests": meta.get("RepoDigests", [])}
        atomic_json(target / "image-pull-progress.json", images)
    atomic_json(path, images)
    return images


def _prepare_enroot_images(cfg, target, rows, path, settings):
    if path.exists():
        frozen = read_json(path)
        for info in frozen.values():
            sqsh = Path(info["sqsh"])
            if not sqsh.is_file() or file_sha256(sqsh) != info["id"]:
                raise StudyError("A frozen container image changed")
        return frozen
    images = {}
    for row in rows:
        output = Path(settings["enroot_data"]) / f"{safe_id(row['instance_id'])}.sqsh"
        import_settings = dict(settings)
        import_settings["image_pull_timeout_seconds"] = cfg["runtime"].get("image_pull_timeout_seconds", 3600)
        images[row["instance_id"]] = enroot_import(import_settings, row["image_tag"], output)
        atomic_json(target / "image-pull-progress.json", images)
    atomic_json(path, images)
    return images


def forbidden_paths(paths):
    result = []
    for path in paths:
        p = PurePosixPath(path)
        if any(part.lower() in ("test", "tests", "evaluation") for part in p.parts) or p.name.startswith("test_") or p.name.endswith("_test.py") or p.name in ("test.py", "tests.py", "conftest.py", "eval.sh", "pytest.ini", "tox.ini", ".coveragerc"):
            result.append(path)
    return result


def _fit_history(backend, messages, output_budget, context_limit, deadline):
    messages = copy.deepcopy(messages)
    while True:
        left = deadline - time.monotonic()
        if left <= 0:
            raise StudyError("wall_time_limit")
        count = backend.count_tokens(messages, timeout=min(30, left))
        room = context_limit - count
        if room > 0 and output_budget <= room:
            return messages, count
        # The prompt fits, so a reply may use every remaining context token.
        if room > 0 and len(messages) <= 1:
            return messages, count
        if len(messages) <= 1:
            raise StudyError("task_exceeds_context_budget")
        del messages[1:3]  # Whole assistant/observed-feedback pairs only.
        notice = "\n[Earlier exchanges omitted to fit the fixed context budget.]"
        if not messages[0]["content"].endswith(notice):
            messages[0]["content"] += notice


def generate_attempt(cfg, row, role, index, image, directory, backend, env_factory=DockerSession):
    directory.mkdir(parents=True, exist_ok=True)
    events = directory / "events.jsonl"
    limits = cfg["rollout_limits"]
    model = cfg["models"][role]
    prompt = (Path(__file__).parent / "prompts/actor.txt").read_text(encoding="utf-8").replace("{{ problem_statement }}", row["problem_statement"])
    messages = [{"role": "user", "content": prompt}]
    stats = {k: 0 for k in ("model_calls", "input_tokens", "output_tokens", "reasoning_tokens_reported", "executed_actions", "format_errors")}
    started = time.monotonic()
    deadline = started + limits["active_rollout_timeout_seconds"]
    env, patch, paths, reason = None, "", [], "step_limit"
    errors, interrupted = 0, False
    atomic_json(directory / "started.json", {"instance_id": row["instance_id"], "generator": role, "candidate_index": index, "time_unix": time.time(), "image": image})
    try:
        image_ref = image.get("sqsh") if cfg["runtime"]["docker"].get("runtime") == "enroot" else image["id"]
        env = env_factory(image_ref, row["base_commit"], cfg["runtime"]["docker"], ttl=limits["active_rollout_timeout_seconds"] + 180)
        for step in range(limits["decision_steps"]):
            left = deadline - time.monotonic()
            if left <= 0:
                reason = "wall_time_limit"
                break
            remaining_tokens = limits["max_total_generated_tokens_per_rollout"] - stats["output_tokens"]
            if remaining_tokens <= 0:
                reason = "output_token_limit"
                break
            allowance = min(remaining_tokens, limits["max_new_tokens_per_action"])
            messages, input_count = _fit_history(backend, messages, allowance, limits["generator_max_context_tokens"], deadline)
            allowance = min(allowance, limits["generator_max_context_tokens"] - input_count)
            if allowance <= 0:
                raise StudyError("task_exceeds_context_budget")
            seed = seed_for(cfg["candidate_pool"]["study_seed"], row["instance_id"], role, index, step)
            event(events, {"type": "model_reservation", "step": step, "input_tokens_counted": input_count, "output_tokens_upper_bound": allowance, "seed": seed})
            stats["model_calls"] += 1
            result = backend.generate(messages, seed=seed, max_new_tokens=allowance, timeout=max(.001, min(limits["model_request_timeout_seconds"], deadline-time.monotonic())))
            event(events, {"type": "model_response", "step": step, "result": result})
            stats["input_tokens"] += result["input_tokens"]
            stats["output_tokens"] += result["output_tokens"]
            stats["reasoning_tokens_reported"] += result.get("reasoning_tokens") or 0
            if result["output_tokens"] > allowance:
                raise StudyError("provider_exceeded_output_budget")
            if time.monotonic() >= deadline:
                reason = "wall_time_limit"
                break
            if result["finish_reason"] == "length":
                reason = "generation_truncated_before_action"
                break
            content = result["content"]
            assistant = {"role": "assistant", "content": content}
            if model.get("chat_template_kwargs", {}).get("preserve_thinking") and result.get("reasoning"):
                assistant["reasoning_content"] = result["reasoning"]
                assistant["reasoning"] = result["reasoning"]
            command = parse_action(content)
            if command is None:
                errors += 1
                stats["format_errors"] += 1
                messages += [assistant, {"role": "user", "content": "FORMAT_ERROR: The final response was not executed. Return exactly one complete bash fenced block in final content, with no surrounding prose. Native thinking stays separate."}]
                if errors >= limits.get("max_consecutive_format_errors", 3):
                    reason = "repeated_format_error"
                    break
                continue
            errors = 0
            # Only a parsed final response can cross the container boundary.
            observed = env.execute(command, min(limits["command_timeout_seconds"], max(1, deadline-time.monotonic())))
            stats["executed_actions"] += 1
            event(events, {"type": "execution", "step": step, "command": command, "command_sha256": hashlib.sha256(command.encode()).hexdigest(), "observation": observed})
            messages += [assistant, {"role": "user", "content": "OBSERVATION:\n" + canonical({**observed, "output": clip(observed["output"], 10000)})}]
            if command.strip() == COMPLETE and observed["returncode"] == 0 and observed["output"].splitlines()[:1] == ["COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"]:
                reason = "submitted"
                break
    except BackendError:
        reason = "infrastructure_model_request_error"
        event(events, {"type": "model_request_unresolved", "step": locals().get("step"), "output_tokens_upper_bound": locals().get("allowance"), "reason": reason})
    except StudyError as error:
        reason = str(error) if str(error) in ("wall_time_limit", "task_exceeds_context_budget", "provider_exceeded_output_budget") else "infrastructure_environment_error"
    except KeyboardInterrupt:
        reason, interrupted = "interrupted", True
    finally:
        if env is not None:
            try:
                patch, paths = env.extract()
            except Exception:
                reason = "infrastructure_patch_extraction_error"
            finally:
                try:
                    env.close()
                except Exception:
                    event(events, {"type": "cleanup_error"})
        invalid = forbidden_paths(paths)
        result = {"instance_id": row["instance_id"], "generator": role, "candidate_index": index, "termination_reason": reason,
                  "model_patch": patch, "patch_sha256": hashlib.sha256(patch.encode()).hexdigest(), "changed_paths": paths,
                  "protocol_status": "invalid_test_or_evaluation_changes" if invalid else "valid", "forbidden_paths": invalid,
                  "stats": stats, "seconds": time.monotonic()-started, "image_id": image["id"], "base_commit": row["base_commit"]}
        (directory / "patch.diff").write_text(patch, encoding="utf-8", newline="\n")
        atomic_json(directory / "result.json", result)
    if interrupted:
        raise KeyboardInterrupt()
    return result


def _candidate_dir(run, role, iid, index):
    return run / "pools" / role / safe_id(iid) / f"candidate_{index}"


def load_candidate(run, role, iid, index):
    directory = _candidate_dir(run, role, iid, index)
    result = read_json(directory / "result.json")
    if hashlib.sha256(result["model_patch"].encode()).hexdigest() != result["patch_sha256"]:
        raise StudyError("Candidate patch hash mismatch")
    return result


def _generator_backend(cfg, role):
    model = cfg["models"][role]
    service = cfg["runtime"]["generator_services"][role]
    if model.get("backend") == "ollama":
        return OllamaGenerator(model, service)
    return VLLMGenerator(model, service)


def generate(cfg, workspace, stage, role, backend=None, env_factory=None):
    require_execution(cfg)
    require_linux()
    run = freeze_run(cfg, workspace, stage)
    rows = read_json(workspace / "data" / stage / "public.json")
    images = read_json(workspace / "data" / stage / "images.json")
    deployment = managed_service(cfg, workspace, role) if backend is None else {"verification": "injected test backend"}
    backend = backend or _generator_backend(cfg, role)
    if env_factory is None:
        env_factory = EnrootSession if cfg["runtime"]["docker"].get("runtime") == "enroot" else DockerSession
    with run_lock(run / ".phase.lock"), Heartbeat(run, "generate", generator=role) as heartbeat:
        identity = backend.verify()
        atomic_json(run / f"service-{role}.json", {"endpoint": identity, **deployment})
        for row in rows:
            for index in range(cfg["candidate_pool"]["main_n"]):
                iid = row["instance_id"]
                target = _candidate_dir(run, role, iid, index)
                heartbeat.data.update(instance_id=iid, candidate_index=index)
                if (target / "result.json").exists():
                    load_candidate(run, role, iid, index)
                    continue
                attempts = sorted(target.glob("attempt_*"))
                max_attempts = cfg["rollout_limits"].get("transport_attempts", 2)
                result, accepted_attempt = None, None
                # Promote an already completed attempt before considering a retry.
                for attempt in attempts:
                    if (attempt / "result.json").exists():
                        saved = read_json(attempt / "result.json")
                        if not saved["termination_reason"].startswith("infrastructure_") and saved["termination_reason"] != "interrupted":
                            result, accepted_attempt = saved, attempt.name
                            break
                if result is None:
                    for number in range(len(attempts)+1, max_attempts+1):
                        attempt = target / f"attempt_{number:04d}"
                        result = generate_attempt(cfg, row, role, index, images[iid], attempt, backend, env_factory)
                        accepted_attempt = attempt.name
                        if not result["termination_reason"].startswith("infrastructure_"):
                            break
                if result is None:
                    completed = [p / "result.json" for p in attempts if (p / "result.json").exists()]
                    result = read_json(completed[-1]) if completed else {"instance_id": iid, "generator": role, "candidate_index": index, "termination_reason": "infrastructure_external_interruption", "model_patch": "", "patch_sha256": hashlib.sha256(b"").hexdigest(), "changed_paths": [], "protocol_status": "valid", "stats": {}, "seconds": 0, "image_id": images[iid]["id"], "base_commit": row["base_commit"]}
                    accepted_attempt = completed[-1].parent.name if completed else None
                result["accepted_attempt"] = accepted_attempt
                result["attempt_count"] = len(list(target.glob("attempt_*")))
                atomic_json(target / "result.json", result)
                (target / "patch.diff").write_text(result["model_patch"], encoding="utf-8", newline="\n")
                print(canonical({"event": "candidate_complete", "generator": role, "instance_id": iid, "candidate_index": index, "reason": result["termination_reason"]}), flush=True)
    return {"run": str(run), "generator": role}


def _pool(run, role, iid, n):
    try:
        return [load_candidate(run, role, iid, i) for i in range(n)]
    except FileNotFoundError:
        raise StudyError("Both complete candidate pools are required before selection/evaluation") from None


def _evidence(run, role, iid, index, budget):
    directory = _candidate_dir(run, role, iid, index)
    accepted = read_json(directory / "result.json").get("accepted_attempt")
    if accepted is None:
        return []
    path = directory / safe_id(accepted) / "events.jsonl"
    if not path.exists():
        return []
    executions = [read for line in path.read_text(encoding="utf-8").splitlines() if (read := json.loads(line)).get("type") == "execution"][-5:]
    if not executions:
        return []
    per_item = max(1, budget // len(executions))
    return [{"command": clip(e["command"], per_item//2), "returncode": e["observation"]["returncode"], "output_excerpt": clip(e["observation"]["output"], per_item//2), "truncated": len(e["command"])+len(e["observation"]["output"]) > per_item} for e in executions]


def select(cfg, workspace, stage, role, selector=None):
    require_execution(cfg)
    require_linux()
    run = freeze_run(cfg, workspace, stage)
    rows = read_json(workspace / "data" / stage / "public.json")
    n = cfg["candidate_pool"]["main_n"]
    generators = active_generators(cfg)
    for gen in generators:
        for row in rows:
            _pool(run, gen, row["instance_id"], n)
    instructions = (Path(__file__).parent / "prompts/selector.txt").read_text(encoding="utf-8")
    gpu = cfg["runtime"]["gpu_roles"][role]
    # Native D3 uses the same CUDA forward-compat libraries recorded for this host.
    prepend_library_path([(cfg["runtime"].get("ollama") or {}).get("cuda_compat_dir")])
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu["visible_devices"]
    selector = selector or NativeD3Selector(cfg["models"][role], gpu, cfg["runtime"].get("hf_cache_dir"))
    pending = []
    for gen in generators:
        for row in rows:
            pool = _pool(run, gen, row["instance_id"], n)
            unique = len({(item["patch_sha256"], item["protocol_status"]) for item in pool})
            pending.append((gen, row, pool, unique))
    with run_lock(run / ".phase.lock"), Heartbeat(run, "select", selector=role):
        try:
            if any(should_verify(unique) for _, _, _, unique in pending):
                atomic_json(run / f"loaded-{role}.json", selector.load())
            for gen, row, pool, unique in pending:
                iid = row["instance_id"]
                path = run / "decisions" / role / gen / f"{iid}.json"
                pool_hash = digest(pool)
                if path.exists():
                    if read_json(path)["pool_sha256"] != pool_hash:
                        raise StudyError("Candidate pool changed after a decision")
                    continue
                indices = list(range(n))
                random.Random(seed_for("presentation", cfg["candidate_pool"]["study_seed"], gen, iid)).shuffle(indices)
                mapping = {f"option_{j+1}": i for j, i in enumerate(indices)}
                criteria = {label: {"diff": pool[i]["model_patch"], "changed_paths": pool[i]["changed_paths"], "protocol_status": pool[i]["protocol_status"], "termination_reason": pool[i]["termination_reason"], "observed_checks": _evidence(run, gen, iid, i, cfg["decision_input"]["observed_output_excerpt_chars_per_candidate"])} for label, i in mapping.items()}
                state = {"issue": row["problem_statement"], "repository": row["repo"], "evidence_policy": "Only recorded command results are observations. Official evaluation labels are unavailable."}
                selected, status, response, failure, choice_seconds = None, "decision_failure", None, None, 0.0
                if not should_verify(unique):
                    selected, status = 0, "identical_pool"
                else:
                    try:
                        choice_started = time.monotonic()
                        response = selector.choose(state, criteria, instructions)
                        selected, status = mapping[response["choice"]], "selected"
                    except (BackendError, KeyError) as error:
                        failure = {"category": "context_error" if "context" in str(error).lower() else "native_backend_error", "reason": str(error), "details": getattr(error, "details", None)}
                    finally:
                        choice_seconds = time.monotonic() - choice_started
                result = {"instance_id": iid, "generator": gen, "selector": role, "pool_sha256": pool_hash, "unique_candidates": unique, "verification": "compared" if should_verify(unique) else "skipped_fewer_than_two_unique", "display_to_candidate": mapping, "selected_index": selected, "status": status, "selected_patch_sha256": pool[selected]["patch_sha256"] if selected is not None else None, "request": {"state": state, "criteria": criteria, "instructions": instructions}, "response": response, "failure": failure, "choice_wall_seconds": choice_seconds}
                atomic_json(path, result)
                print(canonical({"event": "decision_complete", "selector": role, "generator": gen, "instance_id": iid, "status": status}), flush=True)
        finally:
            selector.close()
    return {"run": str(run), "selector": role}


def check_decisions(cfg, workspace, stage):
    run = workspace / "runs" / stage
    rows = read_json(workspace / "data" / stage / "public.json")
    n = cfg["candidate_pool"]["main_n"]
    for judge in active_selectors(cfg):
        for gen in active_generators(cfg):
            for row in rows:
                pool = _pool(run, gen, row["instance_id"], n)
                path = run / "decisions" / judge / gen / f"{row['instance_id']}.json"
                if not path.exists():
                    raise StudyError("Seal the active selectors' decisions before official evaluation")
                decision = read_json(path)
                index = decision["selected_index"]
                if decision["pool_sha256"] != digest(pool) or (index is not None and (index not in range(n) or decision["selected_patch_sha256"] != pool[index]["patch_sha256"])):
                    raise StudyError("Decision/candidate provenance mismatch")
    return rows
