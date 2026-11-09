"""Tool implementations. They run only through the Tool Gateway (kartly.gateway)."""

from kartly.tools.base import CallContext, RiskTier, Tool, ToolEnv, ToolError
from kartly.tools.orders import ORDER_TOOLS
from kartly.tools.writes import WRITE_TOOLS


def default_tools() -> tuple[Tool, ...]:
    """The read-only tools. Write tools (``WRITE_TOOLS``) are added by the caller on purpose."""
    return ORDER_TOOLS


__all__ = [
    "WRITE_TOOLS",
    "CallContext",
    "RiskTier",
    "Tool",
    "ToolEnv",
    "ToolError",
    "default_tools",
]
