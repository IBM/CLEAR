"""Tests for trace_utils normalization functions across provider formats."""

import json

from clear_eval.agentic.pipeline.preprocess_traces.trace_utils import (
    normalize_content,
    _extract_tool_calls_from_content,
    normalize_input_messages,
    extract_from_output_messages,
    normalize_response,
    extract_tool_calls,
    extract_api_spec,
)


# ============================================================================
# normalize_content
# ============================================================================

class TestNormalizeContent:
    """Test text extraction from content blocks across providers."""

    # --- Basic types ---

    def test_none(self):
        assert normalize_content(None) == ""

    def test_string(self):
        assert normalize_content("hello") == "hello"

    def test_dict_json_dumped(self):
        assert normalize_content({"key": "val"}) == '{"key": "val"}'

    def test_other_type(self):
        assert normalize_content(42) == "42"

    # --- OpenAI content blocks ---

    def test_openai_text_block(self):
        blocks = [{"type": "text", "text": "Hello world"}]
        assert normalize_content(blocks) == "Hello world"

    def test_openai_multiple_text_blocks(self):
        blocks = [{"text": "line1"}, {"text": "line2"}]
        assert normalize_content(blocks) == "line1\nline2"

    # --- Anthropic content blocks ---

    def test_anthropic_text_block(self):
        blocks = [{"type": "text", "text": "Hello from Claude"}]
        assert normalize_content(blocks) == "Hello from Claude"

    def test_anthropic_tool_use(self):
        blocks = [{"type": "tool_use", "name": "search", "input": {"query": "weather"}}]
        result = normalize_content(blocks, include_function_calls=True)
        assert "[Tool Call: search(" in result
        assert "weather" in result

    def test_anthropic_tool_use_excluded(self):
        blocks = [{"type": "tool_use", "name": "search", "input": {"query": "weather"}}]
        result = normalize_content(blocks, include_function_calls=False)
        assert result == ""

    def test_anthropic_tool_result_string(self):
        blocks = [{"type": "tool_result", "content": "sunny, 18C"}]
        result = normalize_content(blocks, include_function_calls=True)
        assert "[Tool Result: sunny, 18C]" == result

    def test_anthropic_tool_result_nested_blocks(self):
        blocks = [{"type": "tool_result", "content": [{"type": "text", "text": "result data"}]}]
        result = normalize_content(blocks, include_function_calls=True)
        assert "result data" in result

    def test_anthropic_tool_use_not_confused_with_text(self):
        """tool_use blocks have an 'input' field — should not be treated as text."""
        blocks = [{"type": "tool_use", "name": "calc", "input": {"expr": "2+2"}}]
        result = normalize_content(blocks, include_function_calls=False)
        assert result == ""

    # --- Gemini content blocks ---

    def test_gemini_function_call(self):
        blocks = [{"functionCall": {"name": "get_weather", "args": {"city": "Paris"}}}]
        result = normalize_content(blocks, include_function_calls=True)
        assert "[Tool Call: get_weather(" in result
        assert "Paris" in result

    def test_gemini_function_response(self):
        blocks = [{"functionResponse": {"name": "get_weather", "response": {"temp": 18}}}]
        result = normalize_content(blocks, include_function_calls=True)
        assert "[Tool Result: get_weather" in result

    def test_gemini_inline_data(self):
        blocks = [{"inlineData": {"mimeType": "image/png", "data": "base64..."}}]
        result = normalize_content(blocks)
        assert "[Inline Data: image/png]" == result

    def test_gemini_function_call_excluded(self):
        blocks = [{"functionCall": {"name": "f", "args": {}}}]
        assert normalize_content(blocks, include_function_calls=False) == ""

    # --- Fallback ---

    def test_unknown_dict_block_serialized(self):
        """Unknown dict blocks should be JSON-serialized, not silently dropped."""
        blocks = [{"custom_key": "custom_value"}]
        result = normalize_content(blocks)
        assert "custom_key" in result
        assert "custom_value" in result

    def test_mixed_blocks(self):
        blocks = [
            {"type": "text", "text": "thinking..."},
            {"type": "tool_use", "name": "search", "input": {"q": "test"}},
        ]
        result = normalize_content(blocks, include_function_calls=True)
        assert "thinking..." in result
        assert "[Tool Call: search(" in result

    def test_string_in_list(self):
        assert normalize_content(["hello", "world"]) == "hello\nworld"

    def test_non_string_non_dict_in_list(self):
        assert normalize_content([42, None, "ok"]) == "42\nok"


# ============================================================================
# _extract_tool_calls_from_content
# ============================================================================

class TestExtractToolCallsFromContent:
    """Test the shared helper for extracting tool calls from a message dict."""

    # --- OpenAI ---

    def test_openai_tool_calls(self):
        msg = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "search", "arguments": '{"q": "test"}'}}
            ],
        }
        result = _extract_tool_calls_from_content(msg)
        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["id"] == "call_1"
        assert result[0]["function"]["name"] == "search"
        assert json.loads(result[0]["function"]["arguments"]) == {"q": "test"}

    def test_openai_arguments_already_dict(self):
        """Some providers return arguments as dict instead of string — should be serialized to JSON string."""
        msg = {
            "tool_calls": [{"function": {"name": "f", "arguments": {"x": 1}}}]
        }
        result = _extract_tool_calls_from_content(msg)
        assert result[0]["function"]["arguments"] == json.dumps({"x": 1})

    def test_openai_arguments_invalid_json_string(self):
        """If arguments is a non-JSON string, keep it as-is."""
        msg = {
            "tool_calls": [{"function": {"name": "f", "arguments": "not json"}}]
        }
        result = _extract_tool_calls_from_content(msg)
        assert result[0]["function"]["arguments"] == "not json"

    def test_openai_multiple_tool_calls(self):
        msg = {
            "tool_calls": [
                {"function": {"name": "a", "arguments": "{}"}},
                {"function": {"name": "b", "arguments": "{}"}},
            ]
        }
        result = _extract_tool_calls_from_content(msg)
        assert len(result) == 2
        assert result[0]["function"]["name"] == "a"
        assert result[1]["function"]["name"] == "b"

    def test_openai_flat_tool_call_no_function_wrapper(self):
        """Some Langfuse traces store tool_calls without the function wrapper."""
        msg = {
            "tool_calls": [{"name": "search", "arguments": {"q": "test"}}]
        }
        result = _extract_tool_calls_from_content(msg)
        assert result[0]["function"]["name"] == "search"

    # --- Anthropic ---

    def test_anthropic_tool_use_in_content(self):
        msg = {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Let me search."},
                {"type": "tool_use", "id": "tu_1", "name": "search", "input": {"query": "weather"}},
            ],
        }
        result = _extract_tool_calls_from_content(msg)
        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["id"] == "tu_1"
        assert result[0]["function"]["name"] == "search"
        assert json.loads(result[0]["function"]["arguments"]) == {"query": "weather"}

    def test_anthropic_multiple_tool_use(self):
        msg = {
            "content": [
                {"type": "tool_use", "name": "a", "input": {}},
                {"type": "tool_use", "name": "b", "input": {"x": 1}},
            ]
        }
        result = _extract_tool_calls_from_content(msg)
        assert len(result) == 2

    # --- Gemini ---

    def test_gemini_function_call_in_parts(self):
        msg = {
            "role": "model",
            "parts": [
                {"functionCall": {"name": "get_weather", "args": {"city": "Paris"}}}
            ],
        }
        result = _extract_tool_calls_from_content(msg)
        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "get_weather"
        assert json.loads(result[0]["function"]["arguments"]) == {"city": "Paris"}

    # --- Edge cases ---

    def test_empty_message(self):
        assert _extract_tool_calls_from_content({}) == []

    def test_no_tool_calls(self):
        msg = {"role": "user", "content": "hello"}
        assert _extract_tool_calls_from_content(msg) == []

    def test_string_content_no_crash(self):
        msg = {"role": "assistant", "content": "just text"}
        assert _extract_tool_calls_from_content(msg) == []


# ============================================================================
# normalize_input_messages
# ============================================================================

class TestNormalizeInputMessages:
    """Test input message normalization across providers."""

    # --- Basic types / fallbacks ---

    def test_none(self):
        assert normalize_input_messages(None) == []

    def test_string_passthrough(self):
        assert normalize_input_messages("raw prompt") == "raw prompt"

    def test_non_list_non_dict(self):
        assert normalize_input_messages(42) == "42"

    def test_single_dict_wrapped(self):
        result = normalize_input_messages({"role": "user", "content": "hi"})
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["role"] == "user"

    # --- OpenAI format ---

    def test_openai_basic_conversation(self):
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        result = normalize_input_messages(messages)
        assert len(result) == 3
        assert result[0]["role"] == "system"
        assert result[1]["role"] == "user"
        assert result[2]["role"] == "assistant"
        assert all(r["tool_calls"] == [] for r in result)

    def test_openai_assistant_with_tool_calls(self):
        """Assistant message with null content and tool_calls should NOT be dropped."""
        messages = [
            {"role": "user", "content": "What's the weather?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'}}
                ],
            },
            {"role": "tool", "content": '{"temp": 18}', "tool_call_id": "call_1"},
        ]
        result = normalize_input_messages(messages)
        # Assistant message should be kept (has tool_calls even though content is null)
        assistant_msgs = [m for m in result if m["role"] == "assistant"]
        assert len(assistant_msgs) == 1
        assert len(assistant_msgs[0]["tool_calls"]) == 1
        assert assistant_msgs[0]["tool_calls"][0]["function"]["name"] == "get_weather"

    def test_openai_tool_calls_not_in_content_text(self):
        """Tool calls should be in tool_calls field, not duplicated in content text."""
        messages = [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Let me check."},
                ],
                "tool_calls": [{"function": {"name": "search", "arguments": "{}"}}],
            }
        ]
        result = normalize_input_messages(messages)
        assert result[0]["content"] == "Let me check."
        assert len(result[0]["tool_calls"]) == 1

    # --- Anthropic format ---

    def test_anthropic_messages(self):
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "Hello Claude"}]},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "I'll search for that."},
                    {"type": "tool_use", "id": "tu_1", "name": "search", "input": {"q": "test"}},
                ],
            },
        ]
        result = normalize_input_messages(messages)
        assert len(result) == 2
        assert result[0]["content"] == "Hello Claude"
        assert result[1]["content"] == "I'll search for that."
        assert len(result[1]["tool_calls"]) == 1
        assert result[1]["tool_calls"][0]["function"]["name"] == "search"

    def test_anthropic_tool_use_only_message(self):
        """Anthropic message with only tool_use blocks (no text) should not be dropped."""
        messages = [
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "name": "calc", "input": {"expr": "2+2"}},
                ],
            }
        ]
        result = normalize_input_messages(messages)
        assert len(result) == 1
        assert result[0]["tool_calls"][0]["function"]["name"] == "calc"

    # --- Gemini format ---

    def test_gemini_messages_with_parts(self):
        messages = [
            {"role": "user", "parts": [{"text": "What's the weather?"}]},
            {
                "role": "model",
                "parts": [
                    {"functionCall": {"name": "get_weather", "args": {"city": "Paris"}}}
                ],
            },
        ]
        result = normalize_input_messages(messages)
        assert len(result) == 2
        assert result[0]["content"] == "What's the weather?"
        assert result[1]["tool_calls"][0]["function"]["name"] == "get_weather"

    # --- System message truncation ---

    def test_system_truncation(self):
        messages = [{"role": "system", "content": "x" * 100}]
        result = normalize_input_messages(messages, system_trunc_limit=50)
        assert len(result[0]["content"]) <= 50 + 20  # allow for truncation marker

    # --- Edge cases ---

    def test_none_messages_skipped(self):
        result = normalize_input_messages([None, {"role": "user", "content": "hi"}, None])
        assert len(result) == 1

    def test_string_message_in_list(self):
        result = normalize_input_messages(["raw string message"])
        assert result[0]["role"] == "unknown"
        assert result[0]["content"] == "raw string message"

    def test_empty_content_no_tool_calls_dropped(self):
        """Messages with empty content AND no tool calls should be dropped."""
        messages = [
            {"role": "assistant", "content": ""},
            {"role": "user", "content": "hi"},
        ]
        result = normalize_input_messages(messages)
        assert len(result) == 1
        assert result[0]["role"] == "user"


# ============================================================================
# extract_from_output_messages
# ============================================================================

class TestExtractFromOutputMessages:
    """Test output message extraction across providers."""

    def test_none(self):
        assert extract_from_output_messages(None) == ("", [])

    def test_string(self):
        assert extract_from_output_messages("hello") == ("hello", [])

    def test_non_list(self):
        text, tc = extract_from_output_messages(42)
        assert text == "42"
        assert tc == []

    def test_openai_output_message(self):
        messages = [{"role": "assistant", "content": "The answer is 42."}]
        text, tc = extract_from_output_messages(messages)
        assert text == "The answer is 42."
        assert tc == []

    def test_openai_with_tool_calls(self):
        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"function": {"name": "search", "arguments": '{"q": "test"}'}}
                ],
            }
        ]
        text, tc = extract_from_output_messages(messages)
        assert len(tc) == 1
        assert tc[0]["function"]["name"] == "search"

    def test_anthropic_output_message(self):
        messages = [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Here's what I found."},
                    {"type": "tool_use", "name": "search", "input": {"q": "test"}},
                ],
            }
        ]
        text, tc = extract_from_output_messages(messages)
        assert "Here's what I found." in text
        assert len(tc) == 1
        assert tc[0]["function"]["name"] == "search"

    def test_gemini_output_message(self):
        messages = [
            {
                "role": "model",
                "parts": [
                    {"text": "Let me check."},
                    {"functionCall": {"name": "get_weather", "args": {"city": "Paris"}}},
                ],
            }
        ]
        text, tc = extract_from_output_messages(messages)
        assert "Let me check." in text
        assert len(tc) == 1
        assert tc[0]["function"]["name"] == "get_weather"

    def test_single_dict_wrapped(self):
        msg = {"role": "assistant", "content": "hello"}
        text, tc = extract_from_output_messages(msg)
        assert text == "hello"


# ============================================================================
# extract_tool_calls
# ============================================================================

class TestExtractToolCalls:
    """Test tool call extraction from full output objects."""

    def test_non_dict(self):
        assert extract_tool_calls("string") == []
        assert extract_tool_calls(None) == []

    def test_openai_choices(self):
        output = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {"id": "c1", "type": "function", "function": {"name": "search", "arguments": '{"q": "test"}'}}
                        ],
                    }
                }
            ]
        }
        result = extract_tool_calls(output)
        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "search"
        assert json.loads(result[0]["function"]["arguments"]) == {"q": "test"}

    def test_gemini_candidates(self):
        output = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"functionCall": {"name": "get_weather", "args": {"city": "NYC"}}}
                        ]
                    }
                }
            ]
        }
        result = extract_tool_calls(output)
        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "get_weather"
        assert json.loads(result[0]["function"]["arguments"]) == {"city": "NYC"}

    def test_anthropic_direct_output(self):
        output = {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "I'll search."},
                {"type": "tool_use", "id": "tu_1", "name": "search", "input": {"query": "weather"}},
            ],
        }
        result = extract_tool_calls(output)
        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["id"] == "tu_1"
        assert result[0]["function"]["name"] == "search"
        assert json.loads(result[0]["function"]["arguments"]) == {"query": "weather"}

    def test_direct_tool_calls_field(self):
        """Langfuse sometimes stores tool_calls at top level."""
        output = {
            "tool_calls": [
                {"function": {"name": "f1", "arguments": "{}"}}
            ]
        }
        result = extract_tool_calls(output)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "f1"

    def test_no_tool_calls(self):
        output = {"choices": [{"message": {"content": "just text"}}]}
        assert extract_tool_calls(output) == []


# ============================================================================
# normalize_response
# ============================================================================

class TestNormalizeResponse:
    """Test response text extraction from output objects."""

    def test_none(self):
        assert normalize_response(None) == ""

    def test_string(self):
        assert normalize_response("hello") == "hello"

    def test_openai_choices(self):
        output = {"choices": [{"message": {"content": "The answer is 42."}}]}
        assert normalize_response(output) == "The answer is 42."

    def test_openai_parsed(self):
        output = {"choices": [{"message": {"parsed": {"result": 42}}}]}
        result = normalize_response(output)
        assert "42" in result

    def test_gemini_candidates(self):
        output = {
            "candidates": [
                {"content": {"parts": [{"text": "Hello from Gemini"}]}}
            ]
        }
        assert normalize_response(output) == "Hello from Gemini"

    def test_anthropic_direct(self):
        output = {
            "content": [
                {"type": "text", "text": "Hello from Claude"},
            ]
        }
        assert normalize_response(output) == "Hello from Claude"

    def test_anthropic_tool_use_excluded_from_response(self):
        output = {
            "content": [
                {"type": "text", "text": "Let me search."},
                {"type": "tool_use", "name": "search", "input": {"q": "test"}},
            ]
        }
        result = normalize_response(output)
        assert "Let me search." in result
        assert "Tool Call" not in result

    def test_unknown_dict_fallback(self):
        output = {"weird_key": "weird_value"}
        result = normalize_response(output)
        assert "weird_key" in result
        assert "weird_value" in result


# ============================================================================
# extract_api_spec
# ============================================================================

class TestExtractApiSpec:
    """Test tool definition extraction and normalization."""

    def test_non_dict(self):
        assert extract_api_spec("string") == []
        assert extract_api_spec(None) == []

    def test_openai_tools(self):
        input_data = {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "search",
                        "description": "Search the web",
                        "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
                    },
                }
            ]
        }
        result = extract_api_spec(input_data)
        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "search"
        assert result[0]["function"]["description"] == "Search the web"
        assert "properties" in result[0]["function"]["parameters"]

    def test_anthropic_tools_normalized(self):
        """Anthropic input_schema should be mapped to parameters."""
        input_data = {
            "tools": [
                {
                    "name": "search",
                    "description": "Search the web",
                    "input_schema": {"type": "object", "properties": {"q": {"type": "string"}}},
                }
            ]
        }
        result = extract_api_spec(input_data)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "search"
        assert result[0]["function"]["parameters"] == input_data["tools"][0]["input_schema"]

    def test_gemini_function_declarations(self):
        input_data = {
            "tools": [
                {
                    "functionDeclarations": [
                        {
                            "name": "get_weather",
                            "description": "Get weather",
                            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
                        }
                    ]
                }
            ]
        }
        result = extract_api_spec(input_data)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "get_weather"

    def test_legacy_functions_array(self):
        input_data = {
            "functions": [
                {"name": "old_func", "description": "Legacy", "parameters": {}}
            ]
        }
        result = extract_api_spec(input_data)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "old_func"

    def test_output_shape_consistent(self):
        """All formats should produce the same output keys in full OpenAI format."""
        openai = {"tools": [{"type": "function", "function": {"name": "f", "description": "d", "parameters": {}}}]}
        anthropic = {"tools": [{"name": "f", "description": "d", "input_schema": {}}]}
        gemini = {"tools": [{"functionDeclarations": [{"name": "f", "description": "d", "parameters": {}}]}]}

        for input_data in [openai, anthropic, gemini]:
            result = extract_api_spec(input_data)
            assert len(result) == 1
            assert set(result[0].keys()) == {"type", "function"}
            assert result[0]["type"] == "function"
            assert set(result[0]["function"].keys()) == {"name", "description", "parameters"}
