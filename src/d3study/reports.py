from __future__ import annotations
import json
import os
import random
import shlex
import subprocess
import sys
from pathlib import Path

from .core import (GENERATORS, SELECTORS, Heartbeat, StudyError, active_generators, active_selectors, atomic_json, digest,
                   freeze_run, read_json, require_execution, require_linux, run_lock)
from .docker_env import file_sha256, inspect_image
from .workflow import _pool, check_decisions


def _test_spec(row):
    import json as _json
    from swebench.harness.utils import make_test_spec
    from swebench.types import TestSpec
    if row.get("eval_script") and row.get("log_parser"):
        return make_test_spec(row)
    def names(value):
        if isinstance(value, str):
            return _json.loads(value)
        return list(value or [])
    fail_to_pass = names(row.get("FAIL_TO_PASS"))
    pass_to_pass = names(row.get("PASS_TO_PASS"))
    tests = " ".join(shlex.quote(name) for name in [*fail_to_pass, *pass_to_pass])
    safe = row["instance_id"].replace("/", "_")
    # TestSpec.eval_script prepends `set -uxo pipefail`. Turn that back off, then
    # apply the hidden test patch the way the official harness does before pytest.
    script = "\n".join([
        "set +eux",
        "set +o pipefail",
        "cd /testbed",
        f"git apply --verbose /eval_assets/{safe}.test.patch || patch --batch --fuzz=5 -p1 -i /eval_assets/{safe}.test.patch",
        "for candidate in /opt/miniconda3 /opt/miniconda /opt/conda; do",
        "  if [ -f \"$candidate/etc/profile.d/conda.sh\" ]; then",
        "    source \"$candidate/etc/profile.d/conda.sh\"",
        "    conda activate testbed && break",
        "  fi",
        "done",
        "echo '>>>>> Start Test Output'",
        f"python -m pytest -rA {tests}",
        "echo '>>>>> End Test Output'",
    ])
    parser = "parse_log_pydicom" if "pydicom" in row.get("repo", "") else "parse_log_pytest"
    return TestSpec(
        instance_id=row["instance_id"],
        image=row.get("image") or "",
        eval_script_list=script.splitlines(),
        repo=row["repo"],
        version=str(row.get("version") or ""),
        FAIL_TO_PASS=fail_to_pass,
        PASS_TO_PASS=pass_to_pass,
        log_parser=parser,
        eval_type="pass_and_fail",
    )


def _grade_enroot(cfg, workspace, stage, group, model_name, pending, images, attempt):
    from swebench.harness.grading import get_eval_report
    from .docker_env import EnrootSession
    rows = {row["instance_id"]: row for row in read_json(workspace / "data" / stage / "evaluation.json")}
    run_id = read_json(attempt / "started.json")["run_id"]
    for prediction in pending:
        iid = prediction["instance_id"]
        row = rows[iid]
        spec = _test_spec(row)
        safe = iid.replace("/", "_")
        patch_path = attempt / f"{safe}.diff"
        test_patch_path = attempt / f"{safe}.test.patch"
        script_path = attempt / f"{safe}.eval.sh"
        patch_path.write_text(prediction["model_patch"], encoding="utf-8")
        test_patch_path.write_text(row.get("test_patch") or "", encoding="utf-8")
        script_path.write_text(spec.eval_script, encoding="utf-8")
        log_path = attempt / f"{safe}.test.log"
        mount = f"{attempt.resolve()}:/eval_assets"
        session = EnrootSession(images[iid]["sqsh"], row["base_commit"], cfg["runtime"]["docker"])
        try:
            applied = session.execute(
                f"git apply --verbose /eval_assets/{safe}.diff || patch --batch --fuzz=5 -p1 -i /eval_assets/{safe}.diff",
                120, trusted=True, mounts=(mount,),
            )
            if applied["returncode"] == 0:
                observed = session.execute(
                    f"bash /eval_assets/{safe}.eval.sh",
                    cfg["runtime"]["evaluation"].get("timeout_seconds", 1800),
                    trusted=True,
                    mounts=(mount,),
                )
            else:
                observed = {"output": applied["output"], "returncode": applied["returncode"]}
        finally:
            session.close()
        log_path.write_text(observed["output"], encoding="utf-8")
        report = get_eval_report(
            test_spec=spec,
            prediction=prediction,
            test_log_path=str(log_path),
            include_tests_status=False,
        )
        report_dir = group / "logs" / "run_evaluation" / run_id / model_name / iid
        report_dir.mkdir(parents=True, exist_ok=True)
        atomic_json(report_dir / "report.json", report)


def _harness_report(group, model_name, iid):
    # Only accept reports bound to an explicitly recorded evaluation attempt.
    for started in sorted(group.glob("attempt_*/started.json"), reverse=True):
        run_id = read_json(started)["run_id"]
        path = group / "logs" / "run_evaluation" / run_id / model_name / iid / "report.json"
        if path.exists():
            try:
                report = read_json(path).get(iid)
                if isinstance(report, dict) and type(report.get("resolved")) is bool:
                    return report, path
            except (ValueError, OSError):
                pass
    return None, None


def evaluate(cfg, workspace, stage):
    require_execution(cfg)
    require_linux()
    run = freeze_run(cfg, workspace, stage)
    rows = check_decisions(cfg, workspace, stage)
    n = cfg["candidate_pool"]["main_n"]
    settings = cfg["runtime"]["evaluation"]
    images = read_json(workspace / "data" / stage / "images.json")
    exe = cfg["runtime"]["docker"].get("executable", "docker")
    labels_path = run / "evaluation" / "labels.json"
    labels = read_json(labels_path) if labels_path.exists() else {}
    seal = {str(p.relative_to(run)): digest(read_json(p)) for p in sorted((run / "decisions").rglob("*.json"))}
    with run_lock(run / ".phase.lock"), Heartbeat(run, "evaluate"):
        seal_path = run / "evaluation" / "decision_seal.json"
        if seal_path.exists() and read_json(seal_path) != seal:
            raise StudyError("Decisions changed after evaluation started")
        atomic_json(seal_path, seal)
        generators = active_generators(cfg)
        for generator in generators:
            for index in range(n):
                group = run / "evaluation" / f"{generator}-candidate_{index}"
                group.mkdir(parents=True, exist_ok=True)
                model_name = f"d3study_{stage}_{generator}_candidate_{index}"
                predictions = []
                enroot = cfg["runtime"]["docker"].get("runtime") == "enroot"
                for row in rows:
                    iid = row["instance_id"]
                    if enroot:
                        sqsh = Path(images[iid]["sqsh"])
                        if not sqsh.is_file() or file_sha256(sqsh) != images[iid]["id"]:
                            raise StudyError("Evaluator container image drifted from the generation image")
                    elif inspect_image(exe, images[iid]["tag"])["Id"] != images[iid]["id"]:
                        raise StudyError("Evaluator Docker tag drifted from the generation image")
                    candidate = _pool(run, generator, iid, n)[index]
                    key = f"{generator}/{iid}/{index}"
                    if candidate["protocol_status"] != "valid" or not candidate["model_patch"].strip():
                        labels[key] = {"resolved": False, "source": "empty_or_protocol_invalid", "patch_sha256": candidate["patch_sha256"]}
                    else:
                        predictions.append({"instance_id": iid, "model_name_or_path": model_name, "model_patch": candidate["model_patch"]})
                atomic_json(group / "predictions.json", predictions)
                pending = [p for p in predictions if _harness_report(group, model_name, p["instance_id"])[0] is None]
                if pending:
                    attempt_number = len(list(group.glob("attempt_*"))) + 1
                    attempt = group / f"attempt_{attempt_number:04d}"
                    run_id = f"{stage}-{generator}-c{index}-{digest(cfg)[:10]}-a{attempt_number}"
                    atomic_json(attempt / "predictions.json", pending)
                    atomic_json(attempt / "started.json", {"run_id": run_id, "instance_ids": [p["instance_id"] for p in pending]})
                    if cfg["runtime"]["docker"].get("runtime") == "enroot":
                        _grade_enroot(cfg, workspace, stage, group, model_name, pending, images, attempt)
                        atomic_json(attempt / "finished.json", {"exit_code": 0, "runtime": "enroot"})
                        continue
                    command = [settings.get("python") or sys.executable, "-m", "swebench.harness.run_evaluation", "--dataset_name", str(workspace / "data" / stage / "evaluation.json"), "--split", "test", "--predictions_path", str(attempt / "predictions.json"), "--max_workers", str(settings.get("max_workers", 1)), "--timeout", str(settings.get("timeout_seconds", 1800)), "--run_id", run_id, "--report_dir", str(group)]
                    # Evaluation does not need model-provider or HF credentials.
                    env = os.environ.copy()
                    for name in {"HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "OPENAI_API_KEY", *[s.get("api_key_env") for s in cfg["runtime"]["generator_services"].values()]}:
                        if name:
                            env.pop(name, None)
                    atomic_json(attempt / "command.json", command)
                    with (attempt / "harness.log").open("w", encoding="utf-8") as log:
                        result = subprocess.run(command, cwd=group, env=env, stdout=log, stderr=subprocess.STDOUT)
                    atomic_json(attempt / "finished.json", {"exit_code": result.returncode})
                for prediction in predictions:
                    iid = prediction["instance_id"]
                    candidate = _pool(run, generator, iid, n)[index]
                    report, report_path = _harness_report(group, model_name, iid)
                    labels[f"{generator}/{iid}/{index}"] = {"resolved": report["resolved"] if report else None, "source": "official_harness" if report else "evaluation_error", "patch_sha256": candidate["patch_sha256"], "report": str(report_path.relative_to(run)) if report else None}
                atomic_json(labels_path, labels)
        atomic_json(run / "evaluation" / "summary.json", {"candidate_slots": len(rows)*len(generators)*n, "official_or_defined_labels": sum(v["resolved"] is not None for v in labels.values()), "evaluation_errors": sum(v["resolved"] is None for v in labels.values())})
    return {"labels": str(labels_path), "evaluation_errors": sum(v["resolved"] is None for v in labels.values())}


def bootstrap_interval(values, iterations=2000, seed=42):
    if not values:
        return None
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(values, k=len(values))) / len(values) for _ in range(iterations))
    return [means[int(.025*iterations)], means[min(iterations-1, int(.975*iterations))]]


def analyze(cfg, workspace, stage):
    run = freeze_run(cfg, workspace, stage)
    rows = check_decisions(cfg, workspace, stage)
    labels = read_json(run / "evaluation" / "labels.json")
    seal = read_json(run / "evaluation" / "decision_seal.json")
    if any(digest(read_json(run / p)) != sha for p, sha in seal.items()):
        raise StudyError("A sealed decision changed")
    n = cfg["candidate_pool"]["main_n"]
    summary = {"stage": stage, "task_count": len(rows), "n": n, "generators": {}}
    generators = active_generators(cfg)
    selectors = active_selectors(cfg)
    for gen in generators:
        random_scores, first, oracle = [], [], []
        scores = {judge: [] for judge in selectors}
        failures = {judge: 0 for judge in selectors}
        valid_scores = {judge: [] for judge in selectors}
        failure_categories = {judge: {} for judge in selectors}
        mixed = {judge: [] for judge in selectors}
        all_wrong = all_correct = errors = duplicates = 0
        for row in rows:
            iid = row["instance_id"]
            pool = _pool(run, gen, iid, n)
            ys = []
            for i, candidate in enumerate(pool):
                label = labels[f"{gen}/{iid}/{i}"]
                if label["patch_sha256"] != candidate["patch_sha256"]:
                    raise StudyError("Evaluation/candidate patch hash mismatch")
                if label["resolved"] is None:
                    errors += 1
                ys.append(int(label["resolved"] is True))  # ITT; errors separately visible.
            random_scores.append(sum(ys)/n)
            first.append(ys[0])
            oracle.append(max(ys))
            all_wrong += sum(ys) == 0
            all_correct += sum(ys) == n
            duplicates += n-len({p["patch_sha256"] for p in pool})
            for judge in selectors:
                decision = read_json(run / "decisions" / judge / gen / f"{iid}.json")
                selected = decision["selected_index"]
                failures[judge] += selected is None
                score = ys[selected] if selected is not None else 0
                scores[judge].append(score)
                if selected is not None:
                    valid_scores[judge].append(score)
                else:
                    category = (decision.get("failure") or {}).get("category", "unspecified")
                    failure_categories[judge][category] = failure_categories[judge].get(category, 0) + 1
                if 0 < sum(ys) < n:
                    mixed[judge].append(score)
        m = len(rows)
        judges = {}
        for judge in selectors:
            differences = [a-b for a, b in zip(scores[judge], random_scores)]
            judges[judge] = {"resolved_count": sum(scores[judge]), "resolved_rate": sum(scores[judge])/m, "gain_vs_random": sum(differences)/m, "gain_vs_random_bootstrap_95": bootstrap_interval(differences), "decision_failures": failures[judge], "failure_categories": failure_categories[judge], "valid_decision_count": len(valid_scores[judge]), "valid_decision_resolved_rate": sum(valid_scores[judge])/len(valid_scores[judge]) if valid_scores[judge] else None, "valid_decision_metric_note": "Conditional on a returned selection; evaluation errors still score zero and are counted separately", "mixed_pool_correct": sum(mixed[judge]), "mixed_pool_count": len(mixed[judge])}
        comparison = None
        if "decision_large" in selectors and "decision_small" in selectors:
            paired = [large-small for large, small in zip(scores["decision_large"], scores["decision_small"])]
            comparison = {"rate_difference": sum(paired)/m, "bootstrap_95": bootstrap_interval(paired), "large_wins": sum(x>0 for x in paired), "small_wins": sum(x<0 for x in paired)}
        summary["generators"][gen] = {"single_first_rate": sum(first)/m, "random_expected_rate": sum(random_scores)/m, "oracle_rate": sum(oracle)/m, "oracle_interpretation": "full sampled-pool oracle when evaluation_errors=0; otherwise observed-label lower bound", "all_wrong_or_error_pools": all_wrong, "all_correct_pools": all_correct, "duplicate_slots": duplicates, "evaluation_errors": errors, "selectors": judges, "large_minus_small": comparison}
    # Costs include every attempt, including interrupted/retried work.
    cost = {"model_calls_with_response": 0, "input_tokens": 0, "output_tokens_including_reasoning": 0, "generator_request_seconds": 0.0, "unresolved_model_requests": 0, "selector_seconds": 0.0}
    cost["unresolved_reserved_input_tokens"] = 0
    cost["unresolved_reserved_output_token_upper_bound"] = 0
    for path in (run / "pools").glob("*/*/candidate_*/attempt_*/events.jsonl"):
        reservations, responses, explicit_unresolved = {}, set(), []
        for line in path.read_text(encoding="utf-8").splitlines():
            e = json.loads(line)
            if e["type"] == "model_reservation":
                reservations[e["step"]] = e
            elif e["type"] == "model_response":
                responses.add(e["step"])
                r = e["result"]
                cost["model_calls_with_response"] += 1
                cost["input_tokens"] += r["input_tokens"]
                cost["output_tokens_including_reasoning"] += r["output_tokens"]
                cost["generator_request_seconds"] += r["seconds"]
            elif e["type"] == "model_request_unresolved":
                explicit_unresolved.append(e)
        pending = [r for step, r in reservations.items() if step not in responses]
        cost["unresolved_model_requests"] += len(pending)
        cost["unresolved_reserved_input_tokens"] += sum(r["input_tokens_counted"] for r in pending)
        cost["unresolved_reserved_output_token_upper_bound"] += sum(r["output_tokens_upper_bound"] for r in pending)
        cost.setdefault("preflight_or_unreserved_errors", 0)
        cost["preflight_or_unreserved_errors"] += sum(e.get("step") not in reservations for e in explicit_unresolved)
    for path in (run / "decisions").rglob("*.json"):
        decision = read_json(path)
        cost["selector_seconds"] += decision.get("choice_wall_seconds", 0)
    cost["timing_note"] = "Request wall time is not GPU-active seconds. Use profiler/job accounting for GPU seconds; unresolved requests are not priced as zero."
    summary["measured_costs"] = cost
    atomic_json(run / "analysis" / "summary.json", summary)
    selector_names = list(selectors)
    header = "| Generator | Single first | Random expected | " + " | ".join(selector_names) + " | Oracle/observed upper coverage | Eval errors |"
    align = "|---|---:|---:" + "|---:" * len(selector_names) + "|---:|---:|"
    lines = ["# 双向决策监督的 empirical study：轨迹级 Best-of-3", "", f"Stage: {stage}; tasks per generator: {len(rows)}; N={n}.", "", header, align]
    for gen, result in summary["generators"].items():
        rates = " | ".join(f"{result['selectors'][name]['resolved_rate']:.3f}" for name in selector_names)
        lines.append(f"| {gen} | {result['single_first_rate']:.3f} | {result['random_expected_rate']:.3f} | {rates} | {result['oracle_rate']:.3f} | {result['evaluation_errors']} |")
    lines += ["", "配对区间、输赢题数、混合池分母与全部尝试的资源账目见 summary.json。存在评测错误时，Oracle 数值仅是已观察标签覆盖的下界。", "请求耗时不等于 GPU 活跃计算时间；GPU 秒应来自目标设备的分析器或作业计费记录。"]
    (run / "analysis" / "report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return summary
