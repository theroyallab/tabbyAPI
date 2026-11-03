"""Tools may omit description and parameters, as in the OpenAI API (#485)."""

from endpoints.OAI.types.chat_completion import ChatCompletionRequest


def test_tool_without_parameters_is_accepted():
    request = ChatCompletionRequest(
        messages=[{"role": "user", "content": "hi"}],
        tools=[
            {"type": "function", "function": {"name": "get_goal", "description": "Read it"}},
            {"type": "function", "function": {"name": "list_agents"}},
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
                },
            },
        ],
    )

    dumped = request.model_dump()["tools"]
    assert dumped[0]["function"]["parameters"] == {"type": "object", "properties": {}}
    assert dumped[1]["function"]["description"] == ""
    assert dumped[1]["function"]["parameters"] == {"type": "object", "properties": {}}
    assert dumped[2]["function"]["parameters"]["properties"]["city"] == {"type": "string"}
    # Defaults are per instance, not shared
    dumped[0]["function"]["parameters"]["properties"]["x"] = 1
    assert "x" not in dumped[1]["function"]["parameters"]["properties"]
