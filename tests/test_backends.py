"""CPU-safe adapter tests using fake HTTP/model/GPU interfaces only."""

from __future__ import annotations

import copy
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from d3study import backends


REPO = "Qwen/Qwen3.5-9B"
REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
ALIAS = f"{REPO}@{REVISION}"
D3_REPO = "vllm-sr/d3"
D3_REVISION = "cbff3a3cb0b074b6585fe9f747b9f9f5557a0e75"


def generator_config(**overrides):
    config = {
        "repo": REPO,
        "revision": REVISION,
        "thinking": True,
        "chat_template_kwargs": {"enable_thinking": True},
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 0.0,
        "repetition_penalty": 1.0,
    }
    config.update(overrides)
    return config


class FakeResponse:
    def __init__(self, status_code, body):
        self.status_code = status_code
        self.body = body

    def json(self):
        return copy.deepcopy(self.body)


class FakeSession:
    def __init__(self, *, get_responses=(), post_responses=()):
        self.get_responses = list(get_responses)
        self.post_responses = list(post_responses)
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append(("GET", url, kwargs))
        return self.get_responses.pop(0)

    def post(self, url, **kwargs):
        self.calls.append(("POST", url, kwargs))
        return self.post_responses.pop(0)


def successful_completion(model=ALIAS):
    return {
        "id": "cmpl-test",
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "reasoning_content": "Inspect the failing branch first.",
                    "content": "Run the focused test suite.",
                    "tool_calls": [],
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 31,
            "completion_tokens": 12,
            "completion_tokens_details": {"reasoning_tokens": 7},
        },
    }


def verified_generator(session):
    adapter = backends.VLLMGenerator(
        generator_config(reasoning_effort="xhigh"),
        {"base_url": "http://127.0.0.1:8000/v1"},
        session=session,
    )
    adapter.verify()
    return adapter


class GeneratorTests(unittest.TestCase):
    def test_constructor_is_lazy_and_verify_requires_exact_served_alias(self):
        session = FakeSession(get_responses=[FakeResponse(200, {"data": [{"id": ALIAS}]})])
        adapter = backends.VLLMGenerator(generator_config(), {"base_url": "http://localhost:8000/v1"}, session=session)
        self.assertEqual(session.calls, [])
        self.assertEqual(adapter.verify(), {"model": ALIAS, "verified": True})
        method, url, kwargs = session.calls[0]
        self.assertEqual(method, "GET")
        self.assertEqual(url, "http://localhost:8000/v1/models")
        self.assertIs(kwargs["allow_redirects"], False)

        bad = backends.VLLMGenerator(
            generator_config(),
            {"base_url": "http://localhost:8000/v1"},
            session=FakeSession(get_responses=[FakeResponse(200, {"data": [{"id": REPO}]})]),
        )
        with self.assertRaisesRegex(backends.BackendError, "identity"):
            bad.verify()
        with self.assertRaisesRegex(backends.BackendError, "verify"):
            bad.generate([], seed=1, max_new_tokens=1, timeout=5)

    def test_token_count_and_generation_send_thinking_and_separate_fields(self):
        session = FakeSession(
            get_responses=[FakeResponse(200, {"data": [{"id": ALIAS}]})],
            post_responses=[
                FakeResponse(200, {"count": 9, "max_model_len": 32768}),
                FakeResponse(200, successful_completion()),
            ],
        )
        adapter = verified_generator(session)
        messages = [{"role": "user", "content": "Inspect the project."}]
        self.assertEqual(adapter.count_tokens(messages), 9)
        result = adapter.generate(messages, seed=12, max_new_tokens=1024, timeout=30)

        self.assertEqual(result["content"], "Run the focused test suite.")
        self.assertEqual(result["reasoning"], "Inspect the failing branch first.")
        self.assertEqual((result["input_tokens"], result["output_tokens"], result["reasoning_tokens"]), (31, 12, 7))
        self.assertEqual(result["finish_reason"], "stop")
        self.assertEqual(result["raw_response"]["model"], ALIAS)
        self.assertGreaterEqual(result["seconds"], 0)

        count_body = session.calls[1][2]["json"]
        generation_call = session.calls[2]
        generation_body = generation_call[2]["json"]
        self.assertEqual(session.calls[1][1], "http://127.0.0.1:8000/tokenize")
        self.assertEqual(count_body["chat_template_kwargs"], {"enable_thinking": True})
        self.assertEqual(generation_body["chat_template_kwargs"], {"enable_thinking": True})
        self.assertEqual(generation_body["reasoning_effort"], "xhigh")
        self.assertEqual(generation_body["model"], ALIAS)
        self.assertEqual(generation_body["max_tokens"], 1024)
        self.assertEqual(generation_body["repetition_penalty"], 1.0)
        self.assertEqual(generation_body["seed"], 12)
        self.assertNotIn("reasoning_effort", count_body)
        self.assertIs(generation_call[2]["allow_redirects"], False)

    def test_generation_rejects_mismatch_unsafe_or_malformed_output(self):
        cases = [
            (lambda body: body.update(model="other/model"), "identity"),
            (lambda body: body["choices"][0]["message"].update(tool_calls=[{"id": "x"}]), "Tool calls"),
            (lambda body: body["choices"][0]["message"].update(content="<think>do not parse</think>run test"), "thinking markers"),
            (lambda body: body["usage"].update(completion_tokens=-1), "token usage"),
            (lambda body: body["choices"][0].update(finish_reason=None), "finish reason"),
        ]
        for mutate, message in cases:
            with self.subTest(message=message):
                bad = successful_completion()
                mutate(bad)
                session = FakeSession(
                    get_responses=[FakeResponse(200, {"data": [{"id": ALIAS}]})],
                    post_responses=[FakeResponse(200, bad)],
                )
                adapter = verified_generator(session)
                with self.assertRaisesRegex(backends.BackendError, message):
                    adapter.generate([{"role": "user", "content": "x"}], seed=1, max_new_tokens=5, timeout=2)

    def test_never_follows_redirect_or_retries_and_does_not_echo_body(self):
        session = FakeSession(get_responses=[FakeResponse(302, {"secret": "must not leak"})])
        adapter = backends.VLLMGenerator(generator_config(), {"base_url": "http://localhost:8000/v1"}, session=session)
        with self.assertRaises(backends.BackendError) as caught:
            adapter.verify()
        self.assertNotIn("secret", str(caught.exception))
        self.assertEqual(len(session.calls), 1)
        self.assertIs(session.calls[0][2]["allow_redirects"], False)

    def test_tokenize_rejects_missing_count_instead_of_guessing(self):
        session = FakeSession(
            get_responses=[FakeResponse(200, {"data": [{"id": ALIAS}]})],
            post_responses=[FakeResponse(200, {"max_model_len": 100, "tokens": [1, 2]})],
        )
        adapter = verified_generator(session)
        with self.assertRaisesRegex(backends.BackendError, "unverifiable"):
            adapter.count_tokens([{"role": "user", "content": "x"}])

    def test_length_completion_with_null_content_preserves_usage_and_reasoning(self):
        truncated = successful_completion()
        truncated["choices"][0]["message"]["content"] = None
        truncated["choices"][0]["finish_reason"] = "length"
        truncated["usage"]["completion_tokens"] = 12
        truncated["usage"]["completion_tokens_details"]["reasoning_tokens"] = 9
        session = FakeSession(
            get_responses=[FakeResponse(200, {"data": [{"id": ALIAS}]})],
            post_responses=[FakeResponse(200, truncated)],
        )
        adapter = verified_generator(session)
        result = adapter.generate([{"role": "user", "content": "x"}], seed=5, max_new_tokens=12, timeout=2)
        self.assertEqual(result["content"], "")
        self.assertEqual(result["reasoning"], "Inspect the failing branch first.")
        self.assertEqual(result["finish_reason"], "length")
        self.assertEqual(result["output_tokens"], 12)
        self.assertEqual(result["reasoning_tokens"], 9)

    def test_config_requires_native_thinking(self):
        with self.assertRaisesRegex(backends.BackendError, "thinking"):
            backends.VLLMGenerator(
                generator_config(chat_template_kwargs={"enable_thinking": False}),
                {"base_url": "http://localhost:8000/v1"},
            )


def ollama_chat(content="Run the focused test suite.", thinking="Inspect the failing branch first.", done_reason="stop"):
    return {
        "model": "qwen3.5:2b",
        "message": {"role": "assistant", "content": content, "thinking": thinking},
        "done": True,
        "done_reason": done_reason,
        "prompt_eval_count": 19,
        "eval_count": 8,
    }


class OllamaGeneratorTests(unittest.TestCase):
    def _model(self):
        return {
            "backend": "ollama",
            "ollama_model": "qwen3.5:2b",
            "thinking": True,
            "chat_template_kwargs": {"enable_thinking": True},
            "temperature": 0.6,
            "repetition_penalty": 1.0,
        }

    def _service(self):
        return {"base_url": "http://127.0.0.1:11434", "num_ctx": 32768}

    def test_verify_requires_the_configured_tag(self):
        session = FakeSession(get_responses=[FakeResponse(200, {"models": [{"name": "qwen3.5:2b"}]})])
        adapter = backends.OllamaGenerator(self._model(), self._service(), session=session)
        self.assertEqual(adapter.verify(), {"model": "qwen3.5:2b", "verified": True})
        self.assertEqual(session.calls[0][1], "http://127.0.0.1:11434/api/tags")
        missing = backends.OllamaGenerator(
            self._model(),
            self._service(),
            session=FakeSession(get_responses=[FakeResponse(200, {"models": [{"name": "qwen3.5:9b"}]})]),
        )
        with self.assertRaisesRegex(backends.BackendError, "identity"):
            missing.verify()

    def test_preflight_and_generation_keep_thinking_separate(self):
        session = FakeSession(
            get_responses=[FakeResponse(200, {"models": [{"name": "qwen3.5:2b"}]})],
            post_responses=[FakeResponse(200, ollama_chat()), FakeResponse(200, ollama_chat())],
        )
        adapter = backends.OllamaGenerator(self._model(), self._service(), session=session)
        adapter.verify()
        messages = [{"role": "user", "content": "Inspect the project."}]
        self.assertEqual(adapter.count_tokens(messages), 19)
        result = adapter.generate(messages, seed=12, max_new_tokens=1024, timeout=30)
        self.assertEqual(result["content"], "Run the focused test suite.")
        self.assertEqual(result["reasoning"], "Inspect the failing branch first.")
        self.assertEqual(result["finish_reason"], "stop")
        self.assertEqual((result["input_tokens"], result["output_tokens"]), (19, 8))
        preflight = session.calls[1][2]["json"]
        generation = session.calls[2][2]["json"]
        self.assertEqual(preflight["options"]["num_predict"], 1)
        self.assertTrue(preflight["think"])
        self.assertEqual(generation["options"]["num_predict"], 1024)
        self.assertEqual(generation["options"]["num_ctx"], 32768)
        self.assertEqual(generation["options"]["seed"], 12)
        self.assertEqual(generation["options"]["repeat_penalty"], 1.0)
        self.assertNotIn("seed", preflight["options"])

    def test_repeated_thinking_stops_the_stream(self):
        sentence = "The same analysis is being repeated without a command."
        chunks = []
        for _ in range(8):
            chunks.append(json.dumps({"model": "qwen3.5:2b", "message": {"role": "assistant", "content": "", "thinking": sentence + "\n"}, "done": False}))
        response = FakeResponse(200, {})
        response.iter_lines = lambda decode_unicode=False: iter(chunks)
        response.close = lambda: setattr(response, "closed", True)
        session = FakeSession(
            get_responses=[FakeResponse(200, {"models": [{"name": "qwen3.5:2b"}]})],
            post_responses=[response],
        )
        adapter = backends.OllamaGenerator(self._model(), self._service(), session=session)
        adapter.verify()
        result = adapter.generate([{"role": "user", "content": "x"}], seed=1, max_new_tokens=1000, timeout=5)
        self.assertEqual(result["finish_reason"], "repetition")
        self.assertEqual(result["content"], "")
        self.assertTrue(response.closed)
        self.assertTrue(backends.thinking_is_looping((sentence + "\n") * 6))
        self.assertFalse(backends.thinking_is_looping("Look at the source.\nRun the test.\n"))


class BuilderTests(unittest.TestCase):
    def test_vllm_command_uses_pinned_alias_and_explicit_flags(self):
        command = backends.build_vllm_command(
            generator_config(),
            {
                "tensor_parallel_size": 2,
                "gpu_memory_utilization": 0.82,
                "max_model_len": 32768,
                "max_num_seqs": 1,
            },
            "/opt/models/qwen35/snapshot",
        )
        self.assertEqual(command[:3], ["vllm", "serve", "/opt/models/qwen35/snapshot"])
        expected = {
            "--served-model-name": ALIAS,
            "--reasoning-parser": "qwen3",
            "--dtype": "bfloat16",
            "--tensor-parallel-size": "2",
            "--gpu-memory-utilization": "0.82",
            "--max-model-len": "32768",
            "--max-num-seqs": "1",
        }
        for option, value in expected.items():
            with self.subTest(option=option):
                self.assertEqual(command[command.index(option) + 1], value)

    def test_builder_rejects_nonportable_paths(self):
        paths = [r"C:\models\qwen", "/mnt/c/Users/test/model", "/opt/ollama/models/x", "/opt/models/../escape"]
        for path in paths:
            with self.subTest(path=path):
                with self.assertRaises(backends.BackendError):
                    backends.build_vllm_command(generator_config(), {}, path)


class FakeCuda:
    def __init__(self, available=True, bf16=True):
        self.available = available
        self.bf16 = bf16
        self.calls = []

    def is_available(self):
        return self.available

    def device_count(self):
        return 1

    def is_bf16_supported(self):
        return self.bf16

    def get_device_name(self, index):
        self.calls.append(("get_device_name", index))
        return "Mock GPU"

    def synchronize(self, device):
        self.calls.append(("synchronize", str(device)))

    def reset_peak_memory_stats(self, device):
        self.calls.append(("reset_peak_memory_stats", str(device)))

    def memory_allocated(self, device):
        return 123

    def max_memory_allocated(self, device):
        return 456


class FakeD3Model:
    def __init__(
        self,
        repo=D3_REPO,
        *,
        prepared_tokens=99,
        runtime_max_length=None,
        model_max_positions=262144,
        usage_input_tokens=99,
    ):
        self.runtime = types.SimpleNamespace(device="cuda:0", max_length=runtime_max_length)
        self.runtime.prepare_calls = []
        self.prepared_tokens = prepared_tokens
        self.runtime.prepare = self._prepare
        self.runtime.text = lambda state, question: "rendered prompt"
        self.runtime.tokenizer = lambda texts, **kwargs: {
            "input_ids": [[0] * self.prepared_tokens for _ in texts]
        }
        self.config = types.SimpleNamespace(
            text_config=types.SimpleNamespace(max_position_embeddings=model_max_positions)
        )
        self.manifest = {"repo_id": repo, "identity": {"model_sha256": "model-hash"}}
        self.model_name = D3_REPO
        self.received = None
        self.system_one_calls = 0
        self.usage_input_tokens = usage_input_tokens

    def _prepare(self, state, questions):
        self.runtime.prepare_calls.append((state, questions))
        if self.runtime.max_length is not None and self.prepared_tokens > self.runtime.max_length:
            return types.SimpleNamespace(
                sequences={},
                errors={"choice": {"error": "max_length_exceeded"}},
                questions={"choice": types.SimpleNamespace(rendered={"type": "choice"})},
            )
        return types.SimpleNamespace(
            sequences={"choice": [0] * self.prepared_tokens}, errors={}, questions={}
        )

    def system_one(self, *, state, questions):
        self.system_one_calls += 1
        self.received = (state, questions)
        return {
            "model": self.model_name,
            "answers": {
                "choice": {
                    "type": "choice",
                    "choice": "option_2",
                    "probabilities": {"option_1": 0.2, "option_2": 0.8},
                    "confidence": 0.28,
                }
            },
            "usage": {"input_tokens": self.usage_input_tokens, "output_tokens": 0},
        }


def make_native_import_mocks(
    tmp_path,
    *,
    available=True,
    bf16=True,
    prepared_tokens=99,
    runtime_max_length=None,
    model_max_positions=262144,
    usage_input_tokens=99,
):
    cuda = FakeCuda(available=available, bf16=bf16)
    torch = types.SimpleNamespace(cuda=cuda, bfloat16=object())
    snapshot = tmp_path / D3_REVISION
    snapshot.mkdir()
    model = FakeD3Model(
        prepared_tokens=prepared_tokens,
        runtime_max_length=runtime_max_length,
        model_max_positions=model_max_positions,
        usage_input_tokens=usage_input_tokens,
    )
    calls = {}
    downloads = []

    def from_pretrained(path, **kwargs):
        calls["path"] = path
        calls["kwargs"] = kwargs
        return model

    def snapshot_download(**kwargs):
        downloads.append(kwargs)
        return str(snapshot)

    fake_hub = types.SimpleNamespace(snapshot_download=snapshot_download)
    fake_transformers = types.SimpleNamespace(AutoModel=types.SimpleNamespace(from_pretrained=from_pretrained))

    def import_module(name):
        if name == "torch":
            return torch
        if name == "huggingface_hub":
            return fake_hub
        if name == "transformers":
            return fake_transformers
        raise AssertionError(f"unexpected import: {name}")

    patcher = mock.patch.object(backends.importlib, "import_module", side_effect=import_module)
    patcher.start()
    calls["downloads"] = downloads
    return torch, model, calls, snapshot, patcher


def native_config():
    return {"repo": D3_REPO, "revision": D3_REVISION}


def native_gpu(**overrides):
    config = {
        "device": "cuda:0",
        "question_context_tokens": 32768,
        "batch_size": 1,
        "dtype": "bfloat16",
    }
    config.update(overrides)
    return config


class NativeSelectorTests(unittest.TestCase):
    def _with_native_mocks(self, **kwargs):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        path = Path(temporary.name)
        result = make_native_import_mocks(path, **kwargs)
        self.addCleanup(result[4].stop)
        return result

    def test_load_pins_weights_code_and_single_cuda_device(self):
        torch, model, calls, snapshot, _ = self._with_native_mocks()
        selector = backends.NativeD3Selector(
            native_config(),
            native_gpu(visible_devices="3"),
            cache_dir="/cache/hf",
        )
        with mock.patch.dict(backends.os.environ, {"CUDA_VISIBLE_DEVICES": "3"}):
            metadata = selector.load()
        self.assertEqual(metadata["snapshot_directory_sha"], D3_REVISION)
        self.assertEqual(metadata["code_revision"], D3_REVISION)
        self.assertEqual(metadata["manifest_verification"], "full requested from native loader")
        self.assertEqual(metadata["device"], "cuda:0")
        self.assertEqual(metadata["visible_devices_config"], "3")
        self.assertEqual(metadata["requested_question_context_tokens"], 32768)
        self.assertEqual(metadata["native_context_cap_tokens"], 262144)
        self.assertIsNone(metadata["runtime_max_length"])
        self.assertEqual(metadata["model_max_position_embeddings"], 262144)
        self.assertEqual(calls["path"], str(snapshot))
        self.assertEqual(calls["downloads"], [{"repo_id": D3_REPO, "revision": D3_REVISION, "cache_dir": "/cache/hf"}])
        self.assertEqual(
            {key: value for key, value in calls["kwargs"].items() if key != "dtype"},
            {
                "trust_remote_code": True,
                "revision": D3_REVISION,
                "code_revision": D3_REVISION,
                "device": "cuda:0",
                "batch_size": 1,
                "verify": "full",
            },
        )
        self.assertIs(calls["kwargs"]["dtype"], torch.bfloat16)
        self.assertIs(model, selector._model)

    def test_configured_visible_device_mapping_must_be_set_by_caller(self):
        self._with_native_mocks()
        selector = backends.NativeD3Selector(
            native_config(), native_gpu(visible_devices="3")
        )
        with mock.patch.dict(backends.os.environ, {"CUDA_VISIBLE_DEVICES": "0"}):
            with self.assertRaisesRegex(backends.BackendError, "does not match"):
                selector.load()

    def test_refuses_cpu_or_missing_bf16(self):
        _, _, _, _, _ = self._with_native_mocks(available=False)
        selector = backends.NativeD3Selector(native_config(), native_gpu())
        with self.assertRaisesRegex(backends.BackendError, "no CPU fallback"):
            selector.load()

        _, _, _, _, _ = self._with_native_mocks(bf16=False)
        selector = backends.NativeD3Selector(native_config(), native_gpu())
        with self.assertRaisesRegex(backends.BackendError, "bfloat16"):
            selector.load()

    def test_uses_system_one_and_validates_choice_probabilities(self):
        torch, model, _, _, _ = self._with_native_mocks()
        selector = backends.NativeD3Selector(native_config(), native_gpu())
        selector.load()
        result = selector.choose(
            "Issue and complete diff",
            {"option_1": "Patch A", "option_2": "Patch B"},
            "Which final patch is better supported?",
        )
        self.assertEqual(result["choice"], "option_2")
        self.assertEqual(result["probabilities"], {"option_1": 0.2, "option_2": 0.8})
        self.assertEqual(result["usage"], {"input_tokens": 99, "output_tokens": 0})
        self.assertEqual(result["input_tokens"], 99)
        self.assertEqual(result["preflight_input_tokens"], 99)
        self.assertEqual(model.system_one_calls, 1)
        self.assertEqual(result["gpu_metrics"]["peak_memory_allocated_bytes"], 456)
        self.assertEqual(
            model.received[1],
            {
                "choice": {
                    "type": "choice",
                    "instructions": "Which final patch is better supported?",
                    "criteria": {"option_1": "Patch A", "option_2": "Patch B"},
                }
            },
        )
        self.assertTrue(torch.cuda.calls)

    def test_native_preflight_counts_exact_prompt_without_running_inference(self):
        _, model, _, _, _ = self._with_native_mocks(prepared_tokens=321)
        selector = backends.NativeD3Selector(native_config(), native_gpu())
        selector.load()
        count = selector.count_tokens(
            "Issue and complete diff",
            {"option_1": "Patch A", "option_2": "Patch B"},
            "Which final patch is better supported?",
        )
        self.assertEqual(count, 321)
        self.assertEqual(len(model.runtime.prepare_calls), 1)
        self.assertEqual(model.system_one_calls, 0)

    def test_common_cap_cannot_exceed_native_context_cap(self):
        self._with_native_mocks(model_max_positions=32768)
        selector = backends.NativeD3Selector(
            native_config(), native_gpu(question_context_tokens=32769)
        )
        with mock.patch.dict(backends.os.environ, {"CUDA_VISIBLE_DEVICES": "3"}):
            with self.assertRaisesRegex(backends.BackendError, "exceeds.*context limit") as caught:
                selector.load()
        self.assertEqual(caught.exception.details["native_context_cap_tokens"], 32768)
        self.assertEqual(caught.exception.details["requested_question_context_tokens"], 32769)

    def test_runtime_limit_is_minimum_native_cap(self):
        self._with_native_mocks(runtime_max_length=16384)
        selector = backends.NativeD3Selector(
            native_config(), native_gpu(question_context_tokens=16385)
        )
        with mock.patch.dict(backends.os.environ, {"CUDA_VISIBLE_DEVICES": "3"}):
            with self.assertRaisesRegex(backends.BackendError, "exceeds.*context limit") as caught:
                selector.load()
        self.assertEqual(caught.exception.details["native_context_cap_tokens"], 16384)
        self.assertEqual(caught.exception.details["runtime_max_length"], 16384)

    def test_preflight_rejects_over_common_cap_before_inference(self):
        _, model, _, _, _ = self._with_native_mocks(prepared_tokens=32769)
        selector = backends.NativeD3Selector(native_config(), native_gpu())
        selector.load()
        with self.assertRaisesRegex(backends.BackendError, "common context cap") as caught:
            selector.choose(
                "Issue and complete diff",
                {"option_1": "Patch A", "option_2": "Patch B"},
                "Which final patch is better supported?",
            )
        self.assertEqual(caught.exception.details["error"], "question_context_limit")
        self.assertEqual(caught.exception.details["input_tokens"], 32769)
        self.assertEqual(model.system_one_calls, 0)

    def test_native_limit_preflight_counts_over_limit_prompt_without_inference(self):
        _, model, _, _, _ = self._with_native_mocks(
            prepared_tokens=16385, runtime_max_length=16384
        )
        selector = backends.NativeD3Selector(
            native_config(), native_gpu(question_context_tokens=16384)
        )
        selector.load()
        self.assertEqual(
            selector.count_tokens(
                "state", {"option_1": "A", "option_2": "B"}, "Choose"
            ),
            16385,
        )
        with self.assertRaisesRegex(backends.BackendError, "common context cap"):
            selector.choose("state", {"option_1": "A", "option_2": "B"}, "Choose")
        self.assertEqual(model.system_one_calls, 0)

    def test_reported_usage_over_cap_preserves_raw_usage_details(self):
        _, model, _, _, _ = self._with_native_mocks(usage_input_tokens=32769)
        selector = backends.NativeD3Selector(native_config(), native_gpu())
        selector.load()
        with self.assertRaisesRegex(backends.BackendError, "usage exceeded") as caught:
            selector.choose("state", {"option_1": "A", "option_2": "B"}, "Choose")
        self.assertEqual(caught.exception.details["usage"]["input_tokens"], 32769)
        self.assertEqual(
            caught.exception.details["raw_response"]["usage"]["input_tokens"], 32769
        )

    def test_rejects_bad_choice_answers(self):
        cases = [
            ({"choice": "unknown", "probabilities": {"a": 0.5, "b": 0.5}, "confidence": 0.0}, "outside"),
            ({"choice": "a", "probabilities": {"a": 0.4}, "confidence": 0.0}, "every configured"),
            ({"choice": "a", "probabilities": {"a": 0.4, "b": 0.4}, "confidence": 0.0}, "unnormalized"),
            ({"choice": "a", "probabilities": {"a": 0.5, "b": 0.5}, "confidence": float("nan")}, "confidence"),
            ({"error": {"code": "max_length_exceeded"}}, "failed choice"),
        ]
        for answer, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(backends.BackendError, message):
                    backends.NativeD3Selector._validate_choice_answer(answer, {"a": "A", "b": "B"})

    def test_close_releases_loaded_references(self):
        self._with_native_mocks()
        selector = backends.NativeD3Selector(native_config(), native_gpu())
        selector.load()
        selector.close()
        with self.assertRaisesRegex(backends.BackendError, "load"):
            selector.choose("state", {"a": "A", "b": "B"}, "Choose")


if __name__ == "__main__":
    unittest.main()
