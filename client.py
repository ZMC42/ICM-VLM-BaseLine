#!/usr/bin/env python3

import argparse
import sys

import httpx
from openai import APIError, APITimeoutError, OpenAI


def extract_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if getattr(item, "type", None) == "text":
                text_parts.append(item.text)
        return "\n".join(text_parts).strip()
    return ""


def print_final_message(message, show_reasoning: bool) -> None:
    content = extract_text(message.content)
    reasoning = getattr(message, "reasoning", None)

    if isinstance(reasoning, list):
        reasoning = "\n".join(str(item) for item in reasoning)

    if content:
        print(content)
    elif reasoning:
        print(reasoning)
    else:
        print("[empty content]")

    if show_reasoning and reasoning:
        print("\n[reasoning]\n")
        print(reasoning)


def stream_chat(client: OpenAI, args) -> int:
    printed_reasoning_header = False
    printed_anything = False

    try:
        with client.chat.completions.stream(
            model=args.model,
            messages=[{"role": "user", "content": args.prompt}],
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            extra_body={
                "chat_template_kwargs": {
                    "enable_thinking": args.enable_thinking,
                }
            },
        ) as stream:
            for event in stream:
                if event.type == "content.delta":
                    delta = event.delta or ""
                    if delta:
                        sys.stdout.write(delta)
                        sys.stdout.flush()
                        printed_anything = True
                elif event.type == "reasoning.delta":
                    delta = event.delta or ""
                    if delta and args.show_reasoning:
                        if not printed_reasoning_header:
                            if printed_anything:
                                sys.stdout.write("\n")
                            sys.stdout.write("[reasoning]\n")
                            printed_reasoning_header = True
                        sys.stdout.write(delta)
                        sys.stdout.flush()
                        printed_anything = True

            final = stream.get_final_completion()
    except APITimeoutError as exc:
        print(f"Request timed out: {exc}", file=sys.stderr)
        return 1
    except APIError as exc:
        print(f"API request failed: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Request failed: {exc}", file=sys.stderr)
        return 1

    if printed_anything:
        sys.stdout.write("\n")
        sys.stdout.flush()
        return 0

    print_final_message(final.choices[0].message, args.show_reasoning)
    return 0


def non_stream_chat(client: OpenAI, args) -> int:
    try:
        resp = client.chat.completions.create(
            model=args.model,
            messages=[{"role": "user", "content": args.prompt}],
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            extra_body={
                "chat_template_kwargs": {
                    "enable_thinking": args.enable_thinking,
                }
            },
        )
    except APITimeoutError as exc:
        print(f"Request timed out: {exc}", file=sys.stderr)
        return 1
    except APIError as exc:
        print(f"API request failed: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Request failed: {exc}", file=sys.stderr)
        return 1

    print_final_message(resp.choices[0].message, args.show_reasoning)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Call a local vLLM OpenAI-compatible chat endpoint.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model", default="ModelHub/Qwen/Qwen3.5-9B")
    parser.add_argument("--prompt", default="你好，介绍一下你自己。")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--show-reasoning", action="store_true")
    parser.add_argument("--stream", dest="stream", action="store_true")
    parser.add_argument("--no-stream", dest="stream", action="store_false")
    parser.set_defaults(stream=True)
    args = parser.parse_args()

    http_client = httpx.Client(trust_env=False, timeout=args.timeout)
    client = OpenAI(
        api_key=args.api_key,
        base_url=args.base_url,
        timeout=args.timeout,
        http_client=http_client,
    )

    if args.stream:
        return stream_chat(client, args)
    return non_stream_chat(client, args)


if __name__ == "__main__":
    raise SystemExit(main())
