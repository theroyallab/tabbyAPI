"""Incremental splitting of generated text into reasoning/content/tool channels."""

import re
from typing import List, Optional, Tuple

REASONING = "reasoning"
CONTENT = "content"
TOOL = "tool"


class TagStreamParser:
    """
    Splits a stream of generated text into reasoning/content/tool channels by
    scanning for the model's reasoning and tool call tags.

    Text arrives in arbitrary chunks: a tag may be embedded in a larger span
    (multiple tokens can be released at once) or split across chunks (a tag
    isn't necessarily a single token), so unmatched text that ends with a
    partial tag is held back until a later chunk resolves it.

    Whitespace between the end of a reasoning block and the first content is
    held and dropped if no content ever follows.
    """

    def __init__(
        self,
        reasoning_start: Optional[str] = None,
        reasoning_end: Optional[str] = None,
        tool_start: Optional[str] = None,
        tool_end: Optional[str] = None,
        start_in_reasoning: bool = False,
        tool_calls_in_reasoning: bool = True,
    ):
        self.reasoning_start = reasoning_start
        self.reasoning_end = reasoning_end
        self.tool_start = tool_start
        self.tool_end = tool_end
        self.tool_calls_in_reasoning = tool_calls_in_reasoning

        self.in_reasoning = start_in_reasoning
        self.in_tool = False

        # True if any tag was matched during the last feed() call
        self.saw_tag = False

        # Unemitted text that may still complete into a tag
        self._pending = ""

        # Post-reasoning whitespace hold
        self._holding_ws = False
        self._held_ws = ""

        # Longer tags win when two tags match at the same position
        tags = [t for t in (tool_start, tool_end, reasoning_start, reasoning_end) if t]
        tags.sort(key=len, reverse=True)
        self._tags = tags
        self._tag_re = re.compile("|".join(re.escape(t) for t in tags)) if tags else None
        self._max_hold = max((len(t) - 1 for t in tags), default=0)

    @property
    def in_content(self) -> bool:
        """True while text is being routed to the content channel."""
        return not self.in_reasoning and not self.in_tool

    def feed(self, text: str) -> List[Tuple[str, str]]:
        """Consume a chunk of generated text, returning (channel, text) events."""

        self.saw_tag = False
        events = []
        self._pending += text

        while self._pending:
            match = self._tag_re.search(self._pending) if self._tag_re else None
            if match:
                i, j = match.span()
                self._route(self._pending[:i], events)
                self._pending = self._pending[j:]
                self._handle_tag(match[0], events)
                self.saw_tag = True
            else:
                # Hold back any suffix that is a prefix of a tag
                hold = self._partial_tag_len()
                emit_len = len(self._pending) - hold
                self._route(self._pending[:emit_len], events)
                self._pending = self._pending[emit_len:]
                break

        return _merge_events(events)

    def finish(self) -> List[Tuple[str, str]]:
        """Flush held text at the end of generation. Held whitespace is dropped."""

        events = []
        self._route(self._pending, events)
        self._pending = ""
        return _merge_events(events)

    def _partial_tag_len(self) -> int:
        """Length of the longest pending suffix that could still become a tag."""

        limit = min(self._max_hold, len(self._pending))
        for k in range(limit, 0, -1):
            tail = self._pending[-k:]
            for tag in self._tags:
                if tag.startswith(tail):
                    return k
        return 0

    def _route(self, text: str, events: list):
        """Append text to the currently active channel."""

        if not text:
            return

        if self.in_tool:
            events.append((TOOL, text))
        elif self.in_reasoning:
            events.append((REASONING, text))
        else:
            if self._holding_ws:
                if not text.strip():
                    self._held_ws += text
                    return
                text = self._held_ws + text
                self._held_ws = ""
                self._holding_ws = False
            events.append((CONTENT, text))

    def _handle_tag(self, tag: str, events: list):
        """
        Process a state transition. Tool tags are included in the tool channel
        text; reasoning tags are consumed. No nesting is expected, except tool
        calls may occur inside reasoning content.
        """

        if not self.in_tool:
            if tag == self.reasoning_start:
                self.in_reasoning = True
                self._holding_ws = False
                self._held_ws = ""
                return
            if tag == self.reasoning_end:
                self.in_reasoning = False
                self._holding_ws = True
                return

        if self.in_reasoning and not self.in_tool and not self.tool_calls_in_reasoning:
            if tag in (self.tool_start, self.tool_end):
                # Treat tool tags inside reasoning as plain reasoning text
                events.append((REASONING, tag))
                return

        if tag == self.tool_start:
            self.in_tool = True
            events.append((TOOL, tag))
        elif tag == self.tool_end:
            events.append((TOOL, tag))
            self.in_tool = False


class ChannelStreamParser:
    """
    Base for formats that structure the whole response — reasoning, content
    and tool calls — as a sequence of messages `{header}<|message|>{body}`,
    where the header determines the channel and any other structural token
    ends the body and opens the next header. Generation begins after the
    prompt's `<|start|>assistant`, so parsing starts inside a message header.

    Subclasses define the structural tokens (_TOKENS, which must include
    `<|message|>`), the terminator appended to emitted tool messages
    (_TOOL_END) and the header-to-channel mapping (_resolve_channel).

    Tool messages are emitted on the tool channel as
    `{header}<|message|>{body}{_TOOL_END}`, to be parsed with the matching
    tool call format. The token that ends the last message is usually a stop
    token and never arrives in the text; finish() closes an open tool
    message.
    """

    _HEADER = "header"
    _BODY = "body"

    # Subclass interface
    _TOKENS: List[str]
    _TOOL_END: str

    def _resolve_channel(self, header: str) -> str:
        """Map a message header to a channel."""
        raise NotImplementedError

    def __init__(self):
        self._state = self._HEADER
        self._header = ""
        self._channel = None
        self._pending = ""

        # True if any structural token was matched during the last feed() call
        self.saw_tag = False

        self._token_re = re.compile("|".join(re.escape(t) for t in self._TOKENS))
        self._max_hold = max(len(t) for t in self._TOKENS) - 1

    @property
    def in_reasoning(self) -> bool:
        return self._state == self._BODY and self._channel == REASONING

    @property
    def in_tool(self) -> bool:
        return self._state == self._BODY and self._channel == TOOL

    @property
    def in_content(self) -> bool:
        """True while text is being routed to the content channel."""
        return self._state == self._BODY and self._channel == CONTENT

    def feed(self, text: str) -> List[Tuple[str, str]]:
        """Consume a chunk of generated text, returning (channel, text) events."""

        self.saw_tag = False
        events = []
        self._pending += text

        while self._pending:
            match = self._token_re.search(self._pending)
            if match:
                i, j = match.span()
                self._route(self._pending[:i], events)
                self._pending = self._pending[j:]
                self._handle_token(match[0], events)
                self.saw_tag = True
            else:
                # Hold back any suffix that is a prefix of a structural token
                hold = self._partial_token_len()
                emit_len = len(self._pending) - hold
                self._route(self._pending[:emit_len], events)
                self._pending = self._pending[emit_len:]
                break

        return _merge_events(events)

    def finish(self) -> List[Tuple[str, str]]:
        """
        Flush held text at the end of generation. An open tool message is
        terminated: generation stopped at a stop token, which is not part of
        the text.
        """

        events = []
        self._route(self._pending, events)
        self._pending = ""
        if self.in_tool:
            events.append((TOOL, self._TOOL_END))
            self._state = self._HEADER
            self._header = ""
        return _merge_events(events)

    def _partial_token_len(self) -> int:
        """Length of the longest pending suffix that could still become a token."""

        limit = min(self._max_hold, len(self._pending))
        for k in range(limit, 0, -1):
            tail = self._pending[-k:]
            for token in self._TOKENS:
                if token.startswith(tail):
                    return k
        return 0

    def _route(self, text: str, events: list):
        """Append text to the header or the currently active channel."""

        if not text:
            return

        if self._state == self._HEADER:
            self._header += text
        else:
            events.append((self._channel, text))

    def _handle_token(self, token: str, events: list):
        if token == "<|message|>":
            if self._state == self._HEADER:
                self._channel = self._resolve_channel(self._header)
                if self._channel == TOOL:
                    events.append((TOOL, self._header + "<|message|>"))
                self._state = self._BODY
            return

        # Every other structural token begins a new message header. A stray
        # <|start|> or end token inside a header discards it.
        if self._state == self._BODY and self._channel == TOOL:
            events.append((TOOL, self._TOOL_END))
        self._state = self._HEADER
        self._header = ""


class HarmonyStreamParser(ChannelStreamParser):
    """
    Splits generated text in the Harmony format (gpt-oss) into
    reasoning/content/tool channels.

    Message bodies end with `<|end|>` (another message follows), `<|return|>`
    (end of the response) or `<|call|>` (tool call). The header carries the
    channel and, for tool calls, a recipient:

        <|channel|>analysis<|message|>...reasoning...<|end|>
        <|start|>assistant<|channel|>final<|message|>...content...<|return|>
        <|channel|>commentary to=functions.name <|constrain|>json<|message|>{...}<|call|>

    Routing: any message with a recipient -> tool, analysis -> reasoning,
    final and commentary preambles -> content. Tool messages are parsed with
    the "harmony" tool format.
    """

    _TOKENS = ["<|start|>", "<|message|>", "<|end|>", "<|return|>", "<|call|>"]
    _TOOL_END = "<|call|>"

    def _resolve_channel(self, header: str) -> str:
        channel = re.search(r"<\|channel\|>\s*(\w+)", header)
        channel = channel.group(1) if channel else None
        recipient = re.search(r"\bto=\S+", header)

        # A recipient means the message is addressed to a tool, regardless of
        # channel: function calls normally ride the commentary channel, but
        # gpt-oss also emits them on analysis (the Harmony spec calls
        # built-in tools from there)
        if recipient:
            return TOOL
        if channel == "analysis":
            return REASONING
        # "final", commentary preambles, and anything unrecognized
        return CONTENT


class GlimmerStreamParser(ChannelStreamParser):
    """
    Splits generated text in the Muse Glimmer format into
    reasoning/content/tool channels.

    Message bodies end with `<|eom|>` (another message follows) or `<|eot|>`
    (end of the turn). The header carries an optional recipient:

        <|start|>assistant to=self<|message|>...reasoning...<|eom|>
        <|start|>assistant to=user<|message|>...content...<|eot|>
        <|start|>assistant to=tool.name<|message|><atem:function_calls>...<|eot|>

    Routing: `to=self` -> reasoning, `to=user` or no recipient -> content,
    any other recipient -> tool. Tool messages are parsed with the
    "muse_glimmer" tool format.
    """

    _TOKENS = ["<|start|>", "<|message|>", "<|eom|>", "<|eot|>"]
    _TOOL_END = "<|eom|>"

    def _resolve_channel(self, header: str) -> str:
        recipient = re.search(r"\bto=([^\s<]+)", header)
        recipient = recipient.group(1) if recipient else None

        if recipient == "self":
            return REASONING
        if recipient and recipient != "user":
            return TOOL
        # to=user and messages without a recipient
        return CONTENT


def _merge_events(events: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """Merge consecutive events on the same channel."""

    merged = []
    for channel, text in events:
        if merged and merged[-1][0] == channel:
            merged[-1] = (channel, merged[-1][1] + text)
        else:
            merged.append((channel, text))
    return merged
