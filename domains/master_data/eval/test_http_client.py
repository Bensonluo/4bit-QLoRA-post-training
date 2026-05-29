#!/usr/bin/env python3
"""单条测试 http.client 调用 LM Studio 本地模型."""

import json
import time
import http.client


def call_model_http_client(messages, model_name, base_url, max_tokens=2048, think=False):
    """使用 http.client 调用 OpenAI-compatible API."""
    from urllib.parse import urlparse

    parsed = urlparse(base_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)

    body = {
        "model": model_name,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    if think is not None and "qwen3" in model_name.lower():
        body["extra_body"] = {"think": think}

    body_json = json.dumps(body, ensure_ascii=False)
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer ollama",
        "Content-Length": str(len(body_json.encode("utf-8"))),
    }

    t0 = time.time()
    if parsed.scheme == "https":
        conn = http.client.HTTPSConnection(host, port)
    else:
        conn = http.client.HTTPConnection(host, port)

    try:
        conn.request("POST", "/v1/chat/completions", body=body_json, headers=headers)
        response = conn.getresponse()
        data = response.read().decode("utf-8")
        latency = (time.time() - t0) * 1000

        if response.status != 200:
            print(f"HTTP {response.status}: {data[:200]}")
            return None, latency

        resp_json = json.loads(data)
        text = resp_json["choices"][0]["message"]["content"]
        return text.strip(), latency
    finally:
        conn.close()


def main():
    # 测试 product 单条
    test_msg = [
        {"role": "system", "content": "你是一个药品匹配助手。"},
        {"role": "user", "content": "判断以下候选是否匹配：阿莫西林胶囊"},
    ]

    # 测试 1: 通过代理 1234
    print("=== 测试代理 1234 ===")
    for i in range(3):
        text, lat = call_model_http_client(test_msg, "qwen/qwen3.6-27b", "http://127.0.0.1:1234/v1/")
        print(f"  [{i+1}] status={'OK' if text else 'FAIL'} latency={lat:.0f}ms text={text[:60] if text else None}")
        time.sleep(0.5)

    # 测试 2: 直连 60415
    print("\n=== 测试直连 60415 ===")
    for i in range(3):
        text, lat = call_model_http_client(test_msg, "qwen/qwen3.6-27b", "http://127.0.0.1:60415/v1/")
        print(f"  [{i+1}] status={'OK' if text else 'FAIL'} latency={lat:.0f}ms text={text[:60] if text else None}")
        time.sleep(0.5)


if __name__ == "__main__":
    main()
