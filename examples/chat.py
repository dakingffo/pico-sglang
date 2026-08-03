"""Interactive chat client for pico-sglang / mini-sglang server."""

import sys
import json
import urllib.request


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

        body = json.dumps({
            "model": model,
            "messages": messages,
            "max_tokens": 1024,
            "temperature": 0.7,
        }).encode()

        req = urllib.request.Request(
            f"{base_url}/v1/chat/completions",
            data=body,
            headers={"Content-Type": "application/json"},
        )

        try:
            with urllib.request.urlopen(req) as resp:
                data = json.loads(resp.read())
                reply = data["choices"][0]["message"]["content"]
                messages.append({"role": "assistant", "content": reply})
                print(f"\n{reply}\n")
        except Exception as e:
            print(f"[错误] {e}")
            messages.pop()  # remove the failed user message


if __name__ == "__main__":
    if len(sys.argv) > 1:
        chat(*sys.argv[1:])
    else:
        chat()
