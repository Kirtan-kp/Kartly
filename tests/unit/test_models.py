from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from kartly.store.models import (
    Action,
    Customer,
    Message,
    MessageRole,
    Order,
    OrderStatus,
    PaymentMethod,
    Run,
    RunMode,
    RunStep,
    StepKind,
    Ticket,
    TicketStatus,
)


def _customer(cid: str = "C-9001") -> Customer:
    return Customer(id=cid, name="Test User", email=f"{cid.lower()}@example.com")


def _order(customer_id: str, **overrides: Any) -> Order:
    fields: dict[str, Any] = {
        "id": "KRT-1",
        "customer_id": customer_id,
        "status": OrderStatus.PLACED,
        "payment_method": PaymentMethod.PREPAID,
        "placed_at": datetime(2026, 10, 1, tzinfo=UTC),
        "shipping_address": {"city": "Bhopal"},
        "total_paise": 100_000,
        "refunded_paise": 0,
    }
    fields.update(overrides)
    return Order(**fields)


def _ticket_and_run(session: Session) -> tuple[Ticket, Run]:
    session.add(_customer())
    ticket = Ticket(customer_id="C-9001")
    run = Run(mode=RunMode.SHADOW)
    ticket.runs = [run]
    session.add(ticket)
    session.flush()
    return ticket, run


def test_support_records_round_trip(session: Session) -> None:
    session.add(_customer())
    ticket = Ticket(customer_id="C-9001", external_id="ext-1")
    ticket.messages = [
        Message(
            role=MessageRole.AGENT,
            content="Checking that for you.",
            created_at=datetime(2026, 10, 1, 10, 1, tzinfo=UTC),
        ),
        Message(
            role=MessageRole.CUSTOMER,
            content="Where is my order?",
            created_at=datetime(2026, 10, 1, 10, 0, tzinfo=UTC),
        ),
    ]
    run = Run(mode=RunMode.SHADOW)
    run.steps = [
        RunStep(idx=0, kind=StepKind.STAGE, name="intake", input={"len": 18}),
        RunStep(
            idx=1,
            kind=StepKind.LLM_CALL,
            name="classify",
            model="local-small",
            tokens_in=120,
            tokens_out=15,
        ),
    ]
    ticket.runs = [run]
    session.add(ticket)
    session.commit()
    session.expire_all()

    loaded = session.get(Ticket, ticket.id)
    assert loaded is not None
    assert loaded.status is TicketStatus.RECEIVED
    assert [m.role for m in loaded.messages] == [MessageRole.CUSTOMER, MessageRole.AGENT]
    assert [s.idx for s in loaded.runs[0].steps] == [0, 1]
    assert loaded.runs[0].steps[1].tokens_in == 120


def test_idempotency_key_is_unique(session: Session) -> None:
    ticket, run = _ticket_and_run(session)
    args = {"order_id": "KRT-1", "amount_paise": 50_000}
    session.add(
        Action(
            run_id=run.id,
            ticket_id=ticket.id,
            tool_name="issue_refund",
            args=args,
            idempotency_key="k1",
        )
    )
    session.commit()
    session.add(
        Action(
            run_id=run.id,
            ticket_id=ticket.id,
            tool_name="issue_refund",
            args=args,
            idempotency_key="k1",
        )
    )
    with pytest.raises(IntegrityError):
        session.commit()


def test_run_step_index_is_unique_per_run(session: Session) -> None:
    _, run = _ticket_and_run(session)
    session.add(RunStep(run_id=run.id, idx=0, kind=StepKind.STAGE, name="a"))
    session.commit()
    session.add(RunStep(run_id=run.id, idx=0, kind=StepKind.STAGE, name="b"))
    with pytest.raises(IntegrityError):
        session.commit()


def test_refund_cannot_exceed_order_total(session: Session) -> None:
    session.add(_customer())
    session.add(_order("C-9001", total_paise=100_000, refunded_paise=100_001))
    with pytest.raises(IntegrityError):
        session.flush()


def test_ticket_tier_must_be_between_0_and_3(session: Session) -> None:
    session.add(_customer())
    session.add(Ticket(customer_id="C-9001", tier=4))
    with pytest.raises(IntegrityError):
        session.flush()


def test_enum_values_are_enforced_by_the_database(session: Session) -> None:
    with pytest.raises(IntegrityError):
        session.execute(
            text(
                "INSERT INTO customers (id, name, email, tier) "
                "VALUES ('X1', 'n', 'x1@example.com', 'platinum')"
            )
        )


def test_foreign_keys_are_enforced(session: Session) -> None:
    session.add(Ticket(customer_id="C-DOES-NOT-EXIST"))
    with pytest.raises(IntegrityError):
        session.flush()


def test_deleting_a_ticket_removes_its_dependents(session: Session) -> None:
    ticket, run = _ticket_and_run(session)
    session.add(Message(ticket_id=ticket.id, role=MessageRole.CUSTOMER, content="hi"))
    session.add(RunStep(run_id=run.id, idx=0, kind=StepKind.STAGE, name="intake"))
    session.commit()

    session.delete(ticket)
    session.commit()

    for table in ("tickets", "messages", "runs", "run_steps"):
        assert session.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar_one() == 0
