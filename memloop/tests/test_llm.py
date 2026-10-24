import os

import pytest

from memloop.llm import ChatLLM, request_key

URL = os.environ.get("MEMLOOP_LLM_URL")
MODEL = os.environ.get("MEMLOOP_LLM_MODEL", "qwen3-14b")


def test_request_key_is_order_insensitive_for_params():
    m = [{"role": "user", "content": "hi"}]
    assert request_key("x", m, {"a": 1, "b": 2}) == request_key("x", m, {"b": 2, "a": 1})
    assert request_key("x", m, {"a": 1}) != request_key("y", m, {"a": 1})


@pytest.mark.skipif(not URL, reason="no local LLM server")
def test_greedy_calls_repeat_and_log_serves_cache(tmp_path):
    log = tmp_path / "calls.jsonl"
    llm = ChatLLM(MODEL, base_url=URL, log_path=str(log), max_tokens=20)
    a = llm("Reply with exactly: search[electric toothbrush]")
    b = llm("Reply with exactly: search[electric toothbrush]")
    assert a == b == "search[electric toothbrush]"
    assert llm.stats()["replays_differ"] == 0

    cached = ChatLLM(MODEL, base_url=URL, log_path=str(log), max_tokens=20, use_cache=True)
    assert cached("Reply with exactly: search[electric toothbrush]") == a
    assert cached.stats()["cache_hits"] == 1
