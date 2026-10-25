from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict
from sqlalchemy import Engine, event, func, select
from sqlalchemy.orm import Session, sessionmaker

from kartly.gateway import ToolGateway, ToolResult, compact_schema
from kartly.store.models import ActorType, AuditLog, Customer, Order
from kartly.tools import RiskTier, Tool
from tests.support import (
    CONSTRAINT_TOOL,
    EXPLODING_TOOL,
    FIXED_NOW,
    REJECTING_TOOL,
    WRITING_TOOL,
    make_ctx,
)


def audit_rows(factory: sessionmaker[Session]) -> list[AuditLog]:
    with factory() as session:
        return list(session.scalars(select(AuditLog).order_by(AuditLog.id)))


def customer_exists(factory: sessionmaker[Session], customer_id: str) -> bool:
    with factory() as session:
        return session.get(Customer, customer_id) is not None


# ------------------------------------------------------------------ tool specs


def test_tool_specs_are_compact_and_strict(gateway: ToolGateway) -> None:
    specs = gateway.tool_specs()
    assert [s["function"]["name"] for s in specs] == ["get_order", "get_tracking"]
    assert all(s["type"] == "function" for s in specs)
    params = specs[0]["function"]["parameters"]
    assert params["required"] == ["order_id"]
    assert params["additionalProperties"] is False
    assert "customer_id" not in params["properties"]  # identity is never a model argument
    assert '"title"' not in json.dumps(params)


def test_tool_specs_can_be_limited_to_allowed_tools(gateway: ToolGateway) -> None:
    assert [s["function"]["name"] for s in gateway.tool_specs(["get_tracking"])] == ["get_tracking"]
    with pytest.raises(ValueError, match="nonsense"):
        gateway.tool_specs(["get_order", "nonsense"])


def test_compact_schema_keeps_a_property_that_happens_to_be_named_title() -> None:
    schema = {
        "title": "Args",
        "type": "object",
        "properties": {"title": {"title": "Title", "type": "string"}},
    }
    assert compact_schema(schema) == {
        "type": "object",
        "properties": {"title": {"type": "string"}},
    }


# ------------------------------------------------------------- the happy path


def test_success_is_audited(gateway: ToolGateway, seeded_factory: sessionmaker[Session]) -> None:
    ctx = make_ctx("C-0001")
    result = gateway.call(ctx, "get_order", {"order_id": "KRT-100001"})

    assert result.ok
    assert result.data is not None and result.data["order_id"] == "KRT-100001"
    assert result.latency_ms >= 0
    (row,) = audit_rows(seeded_factory)
    assert row.action == "tool.succeeded"
    assert row.actor_type is ActorType.AGENT
    assert row.actor_id == f"run:{ctx.run_id}"
    assert (row.ticket_id, row.run_id) == (ctx.ticket_id, ctx.run_id)
    assert (row.target_type, row.target_id) == ("order", "KRT-100001")
    assert row.details is not None
    assert row.details["tool"] == "get_order"
    assert row.details["args"] == {"order_id": "KRT-100001"}
    assert "data" not in row.details  # results are not copied into the audit log


def test_model_content_is_compact_json(gateway: ToolGateway) -> None:
    ok = gateway.call(make_ctx("C-0001"), "get_order", {"order_id": "KRT-100001"})
    parsed = json.loads(ok.to_model_content())
    assert parsed["ok"] is True and parsed["data"]["order_id"] == "KRT-100001"
    assert ok.to_model_content().startswith('{"ok":true,"data":{"order_id":')  # no pretty-printing

    bad = gateway.call(make_ctx("C-0001"), "get_order", {})
    parsed = json.loads(bad.to_model_content())
    assert parsed["ok"] is False
    assert parsed["error"]["code"] == "invalid_arguments"


# ------------------------------------------------- identity and access control


def test_other_customers_orders_look_nonexistent(
    gateway: ToolGateway, seeded_factory: sessionmaker[Session]
) -> None:
    foreign = gateway.call(make_ctx("C-0002"), "get_order", {"order_id": "KRT-100001"})
    missing = gateway.call(make_ctx("C-0002"), "get_order", {"order_id": "KRT-999999"})

    assert not foreign.ok and not missing.ok
    assert foreign.data is None
    assert foreign.error_code == missing.error_code == "order_not_found"
    assert foreign.message is not None and missing.message is not None
    # Same message once the ID is masked: nothing tells the model which order IDs exist.
    assert foreign.message.replace("KRT-100001", "<id>") == missing.message.replace(
        "KRT-999999", "<id>"
    )

    foreign_row, missing_row = audit_rows(seeded_factory)
    assert foreign_row.action == missing_row.action == "tool.rejected"
    assert foreign_row.details is not None and missing_row.details is not None
    assert foreign_row.details["reason"] == "not_owner"
    assert foreign_row.details["security_event"] == "cross_customer_access"
    assert missing_row.details["reason"] == "no_such_order"
    assert "security_event" not in missing_row.details


@pytest.mark.parametrize("supplied", ["C-0001", "C-0002", "", None, 7])
def test_a_customer_id_argument_is_always_rejected(
    gateway: ToolGateway, seeded_factory: sessionmaker[Session], supplied: Any
) -> None:
    # Even when the value equals the real owner: identity is never taken from arguments.
    result = gateway.call(
        make_ctx("C-0001"), "get_order", {"order_id": "KRT-100001", "customer_id": supplied}
    )
    assert not result.ok and result.data is None
    assert result.error_code == "invalid_arguments"
    assert result.message is not None and "customer_id" in result.message
    (row,) = audit_rows(seeded_factory)
    assert row.details is not None
    assert row.details["security_event"] == "identity_argument_supplied"


def test_unknown_and_disallowed_tools_are_rejected(
    gateway: ToolGateway, seeded_factory: sessionmaker[Session]
) -> None:
    ctx = make_ctx("C-0001")
    unknown = gateway.call(ctx, "delete_everything", {})
    disallowed = gateway.call(
        ctx, "get_tracking", {"order_id": "KRT-100002"}, allowed_tools={"get_order"}
    )

    assert unknown.error_code == disallowed.error_code == "tool_unavailable"
    assert disallowed.message == "Tool 'get_tracking' is not available. Available tools: get_order."
    unknown_row, disallowed_row = audit_rows(seeded_factory)
    assert unknown_row.details is not None and disallowed_row.details is not None
    assert unknown_row.details["reason"] == "unknown_tool"
    assert "security_event" not in unknown_row.details
    assert disallowed_row.details["reason"] == "not_allowed_for_request"
    assert disallowed_row.details["security_event"] == "disallowed_tool_call"


@pytest.mark.parametrize("bad_args", [None, [], "KRT-100001", 5, [{"order_id": "KRT-100001"}]])
def test_arguments_must_be_a_json_object(gateway: ToolGateway, bad_args: Any) -> None:
    result = gateway.call(make_ctx("C-0001"), "get_order", bad_args)
    assert not result.ok
    assert result.error_code == "invalid_arguments"
    assert result.message is not None and "JSON object" in result.message


def test_invalid_values_are_explained_without_echoing_them(
    gateway: ToolGateway, seeded_factory: sessionmaker[Session]
) -> None:
    ctx = make_ctx("C-0001")
    bad_format = gateway.call(ctx, "get_order", {"order_id": "IGNORE-ALL-RULES"})
    missing = gateway.call(ctx, "get_order", {})

    assert bad_format.message is not None and "order_id" in bad_format.message
    assert "IGNORE-ALL-RULES" not in bad_format.message
    assert missing.message is not None and "Field required" in missing.message
    assert all(row.target_id is None for row in audit_rows(seeded_factory))


def test_oversized_arguments_are_truncated_in_the_audit_log(
    gateway: ToolGateway, seeded_factory: sessionmaker[Session]
) -> None:
    gateway.call(make_ctx("C-0001"), "get_order", {"order_id": "KRT-100001", "note": "x" * 50_000})
    (row,) = audit_rows(seeded_factory)
    assert row.details is not None
    assert len(row.details["args"]["note"]) <= 201


# ------------------------------------------------- failures, rollback, auditing


def test_unexpected_errors_roll_back_the_tool_but_are_audited(
    seeded_factory: sessionmaker[Session],
) -> None:
    gateway = ToolGateway(seeded_factory, [EXPLODING_TOOL], clock=lambda: FIXED_NOW)
    result = gateway.call(make_ctx("C-0001"), "explode", {})

    assert not result.ok and result.error_code == "internal_error"
    assert result.message is not None and "kaboom" not in result.message  # no internals leak
    assert not customer_exists(seeded_factory, "C-BOOM")
    (row,) = audit_rows(seeded_factory)
    assert row.action == "tool.failed"
    assert row.details is not None
    assert row.details["error_type"] == "RuntimeError"
    assert "kaboom" in row.details["error"]


def test_rejections_also_roll_back_partial_writes(
    seeded_factory: sessionmaker[Session],
) -> None:
    gateway = ToolGateway(seeded_factory, [REJECTING_TOOL], clock=lambda: FIXED_NOW)
    result = gateway.call(make_ctx("C-0001"), "reject", {})

    assert result.error_code == "not_allowed" and result.message == "That is not allowed."
    assert not customer_exists(seeded_factory, "C-NOPE")
    (row,) = audit_rows(seeded_factory)
    assert row.action == "tool.rejected"
    assert row.details is not None and row.details["reason"] == "policy"


def test_a_database_error_inside_a_tool_leaves_the_gateway_usable(
    seeded_factory: sessionmaker[Session],
) -> None:
    gateway = ToolGateway(seeded_factory, [CONSTRAINT_TOOL, WRITING_TOOL], clock=lambda: FIXED_NOW)
    failed = gateway.call(make_ctx("C-0001"), "violate", {})
    after = gateway.call(make_ctx("C-0001"), "write", {})

    assert failed.error_code == "internal_error"
    assert after.ok
    assert not customer_exists(seeded_factory, "C-DUP1")
    assert customer_exists(seeded_factory, "C-WROTE")
    assert [row.action for row in audit_rows(seeded_factory)] == ["tool.failed", "tool.succeeded"]


def test_a_call_whose_audit_row_cannot_be_written_has_no_effect(
    engine: Engine, seeded_factory: sessionmaker[Session]
) -> None:
    """Fail closed: no audit record, no action."""
    gateway = ToolGateway(seeded_factory, [WRITING_TOOL], clock=lambda: FIXED_NOW)

    def block_audit_inserts(_conn: Any, _cursor: Any, statement: str, *_rest: Any) -> None:
        if statement.lstrip().lower().startswith("insert into audit_log"):
            raise RuntimeError("audit store unavailable")

    event.listen(engine, "before_cursor_execute", block_audit_inserts)
    try:
        with pytest.raises(RuntimeError, match="audit store unavailable"):
            gateway.call(make_ctx("C-0001"), "write", {})
    finally:
        event.remove(engine, "before_cursor_execute", block_audit_inserts)

    assert not customer_exists(seeded_factory, "C-WROTE")
    with seeded_factory() as session:
        assert session.scalar(select(func.count()).select_from(AuditLog)) == 0


def test_read_tools_do_not_change_the_store(
    gateway: ToolGateway, seeded_factory: sessionmaker[Session]
) -> None:
    with seeded_factory() as session:
        before = session.scalar(select(func.sum(Order.refunded_paise)))
    for order_id in ("KRT-100001", "KRT-100002", "KRT-100008"):
        for tool in ("get_order", "get_tracking"):
            gateway.call(make_ctx("C-0001"), tool, {"order_id": order_id})
    with seeded_factory() as session:
        assert session.scalar(select(func.sum(Order.refunded_paise))) == before


# ------------------------------------------------------------- registration


def test_duplicate_tool_names_are_rejected(seeded_factory: sessionmaker[Session]) -> None:
    with pytest.raises(ValueError, match="duplicate"):
        ToolGateway(seeded_factory, [WRITING_TOOL, WRITING_TOOL])


def test_tools_must_forbid_unknown_arguments() -> None:
    class Lenient(BaseModel):
        model_config = ConfigDict(extra="ignore")

    with pytest.raises(ValueError, match="extra='forbid'"):
        Tool(
            name="lenient",
            tier=RiskTier.READ_ONLY,
            description="x",
            args_model=Lenient,
            handler=lambda _env, _args: {},
        )


@pytest.mark.parametrize("name", ["", "Get_Order", "get-order", "1order", "x" * 61])
def test_tool_names_are_validated(name: str) -> None:
    class Args(BaseModel):
        model_config = ConfigDict(extra="forbid")

    with pytest.raises(ValueError, match="invalid tool name"):
        Tool(
            name=name,
            tier=RiskTier.READ_ONLY,
            description="x",
            args_model=Args,
            handler=lambda _env, _args: {},
        )


def test_tool_result_is_immutable() -> None:
    result = ToolResult(ok=True, tool="t")
    with pytest.raises(AttributeError):
        result.ok = False  # type: ignore[misc]
