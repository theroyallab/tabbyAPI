from endpoints.OAI.types.tools import ToolCall
from endpoints.OAI.utils.toolcall_formats.common import FormatSignature
from endpoints.OAI.utils.toolcall_formats import deepseek_v4

"""
DeepSeek-V4.1 - the DSML dialect of V4 with a space after the token and a
shorter outer tag

Raw format:
    <｜DSML｜ calls>
    <｜DSML｜ invoke name="__FUNCTION_NAME__">
    <｜DSML｜ parameter name="__PARAM_1__" string="true|false">__VALUE_1__</｜DSML｜ parameter>
    <｜DSML｜ parameter name="__PARAM_2__" string="true|false">__VALUE_2__</｜DSML｜ parameter>
    </｜DSML｜ invoke>
    <｜DSML｜ invoke name="__FUNCTION_NAME_2__">
    ...
    </｜DSML｜ invoke>
    </｜DSML｜ calls>

Parameter typing is the same as V4 (string="true" verbatim, string="false"
JSON), and the V4 parser accepts both spellings of the invoke and parameter
tags, so parsing is shared. The outer tag is what separates the formats: it is
the marker the stream parser waits for, and the template evidence that tells
the two apart (V4 renders <｜DSML｜tool_calls>, V4.1 <｜DSML｜ calls>). The
architecture name (DeepseekV41) contains V4's, so the template is what decides.

Like V4, the template reorders tool results into tool call order itself, and
the server sorts tool messages before templating anyway.
"""

TOOLCALL_START = "<｜DSML｜ calls>"
TOOLCALL_END = "</｜DSML｜ calls>"

DETECT = FormatSignature(
    template_markers=("<｜DSML｜ calls>",),
    special_tokens=("｜DSML｜",),
    architectures=("DeepseekV41",),
    reasoning_tags=("<think>", "</think>"),
)


def parse_toolcalls(text: str) -> list[ToolCall]:
    return deepseek_v4.parse_toolcalls(text, log_name="deepseek_v41")
