import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from common import model  # noqa: F401 - initialize backend imports before the core router
from endpoints.OAI.types.chat_completion import ChatCompletionMessage
from endpoints.core import router
from endpoints.core.types.token import TokenEncodeRequest


class TokenEncodeRequestTests(unittest.TestCase):
    def test_accepts_chat_template_kwargs_alias(self):
        request = TokenEncodeRequest(
            text="prompt",
            chat_template_kwargs={"enable_thinking": False},
        )

        self.assertEqual(request.template_vars, {"enable_thinking": False})


class TokenEncodeEndpointTests(unittest.IsolatedAsyncioTestCase):
    async def test_formats_messages_and_leaves_bos_handling_to_tokenizer(self):
        formatter = AsyncMock(return_value=("<s>rendered prompt", None, {"bos_token": "<s>"}))
        encode_tokens = Mock(return_value=[1, 2, 3])
        container = SimpleNamespace(prompt_template=object(), encode_tokens=encode_tokens)
        request = TokenEncodeRequest(
            text=[ChatCompletionMessage(content="hello")],
            chat_template_kwargs={
                "add_generation_prompt": True,
                "enable_thinking": False,
            },
            add_bos_token=False,
        )

        with (
            patch.object(router.config.network, "api_servers", ["oai"]),
            patch.object(router.model, "container", container),
            patch.object(router, "format_messages_with_template", formatter),
        ):
            response = await router.encode_tokens(request)

        formatter.assert_awaited_once_with(
            request.text,
            {
                "add_generation_prompt": False,
                "enable_thinking": False,
            },
        )
        encode_tokens.assert_called_once_with(
            "rendered prompt",
            embeddings=None,
            add_bos_token=False,
            encode_special_tokens=True,
            decode_special_tokens=True,
        )
        self.assertEqual(response.tokens, [1, 2, 3])
        self.assertEqual(response.length, 3)


if __name__ == "__main__":
    unittest.main()
