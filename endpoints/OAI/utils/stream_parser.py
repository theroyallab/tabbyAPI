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


def _merge_events(events: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """Merge consecutive events on the same channel."""

    merged = []
    for channel, text in events:
        if merged and merged[-1][0] == channel:
            merged[-1] = (channel, merged[-1][1] + text)
        else:
            merged.append((channel, text))
    return merged
