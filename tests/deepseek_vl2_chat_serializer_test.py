"""Regression tests for DeepSeek-VL2 built-in chat serialization."""

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from endpoints.OAI.utils import chat_completion as cc


class _FakeMultimodalEmbeddingWrapper:
    """Minimal multimodal stub that emits stable text aliases."""

    def __init__(self):
        self.text_alias = []
        self.urls = []

    async def add(self, url: str):
        self.urls.append(url)
        self.text_alias.append(f"<image_{len(self.text_alias) + 1}>")


def _message(role: str, content, tool_call_id=None):
    return SimpleNamespace(role=role, content=content, tool_call_id=tool_call_id)


def _text_part(text: str):
    return SimpleNamespace(type="text", text=text, image_url=None)


def _image_part(url: str):
    return SimpleNamespace(
        type="image_url",
        text=None,
        image_url=SimpleNamespace(url=url),
    )


def _install_deepseek_vl2_serializer(monkeypatch):
    container = SimpleNamespace(
        config=SimpleNamespace(architecture=cc.DEEPSEEK_VL2_ARCH),
        use_vision=True,
    )
    monkeypatch.setattr(cc.model, "container", container, raising=False)
    monkeypatch.setattr(
        cc,
        "MultimodalEmbeddingWrapper",
        _FakeMultimodalEmbeddingWrapper,
    )


def test_builtin_serializer_supports_official_multi_image_interleaving(monkeypatch):
    _install_deepseek_vl2_serializer(monkeypatch)

    messages = [
        _message(
            "user",
            [
                _text_part("This is image_1: "),
                _image_part("image://1"),
                _text_part("This is image_2: "),
                _image_part("image://2"),
                _text_part("This is image_3: "),
                _image_part("image://3"),
                _text_part(" Can you tell me what are in the images?"),
            ],
        ),
        _message("assistant", ""),
    ]

    prompt, mm_embeddings, serializer_state = asyncio.run(
        cc.format_messages_with_builtin_serializer(messages)
    )

    assert prompt == (
        "<|User|>: This is image_1: \n<image_1>\n"
        "This is image_2: \n<image_2>\n"
        "This is image_3: \n<image_3>\n"
        " Can you tell me what are in the images?\n\n"
        "<|Assistant|>: "
    )
    assert mm_embeddings is not None
    assert mm_embeddings.urls == ["image://1", "image://2", "image://3"]
    assert mm_embeddings.text_alias == ["<image_1>", "<image_2>", "<image_3>"]
    assert serializer_state["last_non_system_role"] == "assistant"


def test_builtin_serializer_preserves_grounding_markup(monkeypatch):
    _install_deepseek_vl2_serializer(monkeypatch)

    grounding_text = (
        "<|ref|>The red square<|/ref|> is the target. "
        "<|grounding|><|det|>[0.1,0.2,0.9,0.9]<|/det|><|/grounding|>"
    )
    messages = [
        _message(
            "user",
            [
                _image_part("image://1"),
                _text_part(grounding_text),
            ],
        ),
    ]

    prompt, mm_embeddings, serializer_state = asyncio.run(
        cc.format_messages_with_builtin_serializer(messages)
    )

    assert prompt == f"<|User|>: <image_1>\n{grounding_text}"
    assert mm_embeddings is not None
    assert mm_embeddings.urls == ["image://1"]
    assert serializer_state["last_non_system_role"] == "user"
