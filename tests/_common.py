import json
import httpx
from pprint import pprint


def test_chat_request(api_key, base_url, request, n=1):
    request["n"] = n

    response = httpx.post(
        f"{base_url}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json=request,
        timeout=300,
    )

    data = response.json()

    print("\n\n\n")
    print("--------------------------------------------------------------------------------")
    print(f"REQUEST, n={n}:")
    print("--------------------------------------------------------------------------------")
    print()
    pprint(data, width=160)
    print()

    for idx in range(len(data["choices"])):
        print(f"\n---- choice {idx}\n")
        print(f"\nFinish reason: {data['choices'][idx]['finish_reason']}\n")
        message = data["choices"][idx]["message"]

        if message.get("reasoning_content"):
            print(f"\nreasoning_content:\n{message['reasoning_content']}")

        if message.get("content"):
            print(f"\ncontent:\n{message['content']}")

        if message.get("tool_calls"):
            print()
            for tc in message["tool_calls"]:
                fn = tc["function"]
                args = json.loads(fn["arguments"])
                print(f"  [{tc['id']}] {fn['name']}({json.dumps(args)})")
        else:
            print("\n[No tool calls requested]")

    if "usage" in data:
        print("\n[usage]")
        pprint(data["usage"])


def test_comp_request(api_key, base_url, request, n=1):
    request["n"] = n

    response = httpx.post(
        f"{base_url}/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json=request,
        timeout=300,
    )

    data = response.json()

    print("\n\n\n")
    print("--------------------------------------------------------------------------------")
    print(f"COMPLETIONS REQUEST, n={n}:")
    print("--------------------------------------------------------------------------------")
    print()
    pprint(data, width=160)
    print()

    for idx in range(len(data["choices"])):
        print(f"\n---- choice {idx}\n")
        print(f"\nFinish reason: {data['choices'][idx]['finish_reason']}\n")
        choice = data["choices"][idx]

        if "text" in choice:
            print(f"\ntext:\n{choice['text']}")

    if "usage" in data:
        print("\n[usage]")
        pprint(data["usage"])


def test_chat_streaming(api_key, base_url, request, n=1, display_idx=0, rawdump=False):
    print("\n\n\n")
    print("--------------------------------------------------------------------------------")
    print("STREAMING REQUEST:")
    print("--------------------------------------------------------------------------------")

    mode = None

    request["stream"] = True
    request["n"] = n
    usage = None

    tool_calls = [{} for _ in range(n)]

    with httpx.stream(
        "POST",
        f"{base_url}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json=request,
        timeout=300,
    ) as resp:
        for line in resp.iter_lines():
            if not line.startswith("data: "):
                continue
            payload = line[len("data: ") :]
            if payload == "[DONE]":
                break
            chunk = json.loads(payload)
            if rawdump:
                pprint(chunk, width=160)

            if "usage" in chunk:
                usage = chunk["usage"]

            for choice in chunk["choices"]:
                idx = choice["index"]

                delta = choice["delta"]
                finish_reason = choice.get("finish_reason")

                if delta.get("reasoning_content"):
                    if mode != "reasoning":
                        mode = "reasoning"
                        if idx == display_idx:
                            print(f"\n\n[thinking][{idx}]")
                    if not rawdump and idx == display_idx:
                        print(delta["reasoning_content"], end="", flush=True)

                if delta.get("content"):
                    if mode != "content":
                        mode = "content"
                        if idx == display_idx:
                            print(f"\n\n[content][{idx}]")
                    if not rawdump and idx == display_idx:
                        print(delta["content"], end="", flush=True)

                if delta.get("tool_calls"):
                    if mode != "tool_calls":
                        mode = "tool_calls"
                        if idx == display_idx:
                            print(f"\n\n[tool_calls][{idx}]")
                    for tc in delta["tool_calls"]:
                        tcidx = tc["index"]
                        if tcidx not in tool_calls:
                            tool_calls[idx][tcidx] = {
                                "id": tc.get("id", ""),
                                "name": tc["function"].get("name", ""),
                                "arguments": "",
                            }
                        if tc["function"].get("arguments"):
                            tool_calls[idx][tcidx]["arguments"] += tc["function"]["arguments"]

                if finish_reason:
                    print(f"\nFinish reason [{idx}]: {finish_reason}")

    for ci, tcs in enumerate(tool_calls):
        if tcs:
            print(f"\nTool calls [{ci}]:")
            for idx in sorted(tcs):
                tc = tcs[idx]
                if rawdump:
                    print("args:" + tc["arguments"])
                args = json.loads(tc["arguments"])
                print(f"  [{tc['id']}] {tc['name']}({json.dumps(args)})")
        else:
            print("\n[No tool calls requested]")

    if usage:
        print("\n[usage]")
        pprint(usage)


def test_comp_streaming(api_key, base_url, request, n=1, display_idx=0, rawdump=False):
    print("\n\n\n")
    print("--------------------------------------------------------------------------------")
    print("STREAMING COMPLETIONS REQUEST:")
    print("--------------------------------------------------------------------------------")

    request["stream"] = True
    request["n"] = n
    usage = None

    with httpx.stream(
        "POST",
        f"{base_url}/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json=request,
        timeout=300,
    ) as resp:
        for line in resp.iter_lines():
            if not line.startswith("data: "):
                continue
            payload = line[len("data: ") :]
            if payload == "[DONE]":
                break
            chunk = json.loads(payload)
            if rawdump:
                pprint(chunk, width=160)

            if "usage" in chunk:
                usage = chunk["usage"]

            for choice in chunk["choices"]:
                idx = choice["index"]
                finish_reason = choice.get("finish_reason")

                if choice.get("text"):
                    if not rawdump and idx == display_idx:
                        print(choice["text"], end="", flush=True)

                if finish_reason:
                    print(f"\nFinish reason [{idx}]: {finish_reason}")

    if usage:
        print("\n[usage]")
        pprint(usage)
