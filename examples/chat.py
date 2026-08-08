"""Interactive chat client for pico-sglang / mini-sglang server (streaming)."""

import json
import sys
import urllib.request


def _stream_chat(base_url: str, model: str, messages: list[dict]) -> str:
    """Send a chat request with stream=True, yield + return full reply."""
    body = json.dumps({
        "model": model,
        "messages": messages,
        "max_tokens": 1024,
        "temperature": 0.7,
        "stream": True,
    }).encode()

    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )

    full = ""
    with urllib.request.urlopen(req) as resp:
        for line in resp:
            line = line.decode().strip()
            if not line.startswith("data:"):
                continue
            payload = line[len("data:"):].strip()
            if payload == "[DONE]":
                break
            chunk = json.loads(payload)
            choices = chunk.get("choices")
            if not choices:
                continue
            delta = choices[0].get("delta", {})
            piece = delta.get("content")
            if piece:
                full += piece
                print(piece, end="", flush=True)
    print()
    return full


def chat(base_url: str = "http://127.0.0.1:1919", model: str = "qwen") -> None:
    messages: list[dict] = []
    print("pico-sglang chat (输入 /exit 退出, /clear 清空历史)\n")

    while True:
        try:
            user_input = input(">>> ")
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            break

        if user_input.strip() == "/exit":
            break
        if user_input.strip() == "/clear":
            messages.clear()
            print("[历史已清空]\n")
            continue
        if not user_input.strip():
            continue

        messages.append({"role": "user", "content": user_input})

        try:
            reply = _stream_chat(base_url, model, messages)
            messages.append({"role": "assistant", "content": reply})
        except Exception as e:
            print(f"[错误] {e}")
            messages.pop()  # remove the failed user message


if __name__ == "__main__":
    if len(sys.argv) > 1:
        chat(*sys.argv[1:])
    else:
        chat()
