"""SQLAlchemy models: the mock store plus the support-side tables.

Conventions
- Money is stored as integer paise (1 INR = 100 paise) to avoid float errors.
- Store entities (customers, orders) use readable string IDs so eval fixtures are easy to write.
  Support-side records (tickets, runs, actions, ...) use UUIDs.
- Enums are stored as plain strings with CHECK constraints (portable, easy to migrate).
- JSON columns use JSONB on Postgres and generic JSON elsewhere (SQLite in unit tests).
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    Enum,
    ForeignKey,
    Integer,
    MetaData,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    Uuid,
    false,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)


JSON_TYPE = JSON().with_variant(JSONB(), "postgresql")
BIGINT_PK = BigInteger().with_variant(Integer, "sqlite")


# --------------------------------------------------------------------------- enums


class CustomerTier(enum.StrEnum):
    STANDARD = "standard"
    SILVER = "silver"
    GOLD = "gold"


class OrderStatus(enum.StrEnum):
    PLACED = "placed"
    CONFIRMED = "confirmed"
    PACKED = "packed"
    SHIPPED = "shipped"
    OUT_FOR_DELIVERY = "out_for_delivery"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"


class PaymentMethod(enum.StrEnum):
    PREPAID = "prepaid"
    COD = "cod"


class ShipmentStatus(enum.StrEnum):
    IN_TRANSIT = "in_transit"
    OUT_FOR_DELIVERY = "out_for_delivery"
    DELIVERED = "delivered"
    EXCEPTION = "exception"


class TicketStatus(enum.StrEnum):
    """Mirrors the ticket state machine in DESIGN.md section 5."""

    RECEIVED = "received"
    SCREENED = "screened"
    ROUTED = "routed"
    RESOLVING = "resolving"
    VALIDATING = "validating"
    SENT = "sent"
    DRAFTED = "drafted"
    AWAITING_APPROVAL = "awaiting_approval"
    ESCALATED = "escalated"
    CLOSED = "closed"
    FAILED = "failed"


class MessageRole(enum.StrEnum):
    CUSTOMER = "customer"
    AGENT = "agent"
    HUMAN_AGENT = "human_agent"
    SYSTEM = "system"


class RunStatus(enum.StrEnum):
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class RunMode(enum.StrEnum):
    SHADOW = "shadow"
    AUTO = "auto"


class StepKind(enum.StrEnum):
    STAGE = "stage"
    LLM_CALL = "llm_call"
    TOOL_CALL = "tool_call"


class ActionStatus(enum.StrEnum):
    PENDING = "pending"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    REJECTED = "rejected"
    AWAITING_APPROVAL = "awaiting_approval"


class ApprovalStatus(enum.StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"


class ActorType(enum.StrEnum):
    AGENT = "agent"
    HUMAN = "human"
    SYSTEM = "system"


def _enum(cls: type[enum.StrEnum], name: str) -> Enum:
    return Enum(
        cls,
        name=name,
        native_enum=False,
        length=32,
        create_constraint=True,
        validate_strings=True,
        values_callable=lambda e: [m.value for m in e],
    )


def _created_at() -> Mapped[datetime]:
    return mapped_column(DateTime(timezone=True), server_default=func.now())


# ------------------------------------------------------------------ mock store


class Customer(Base):
    __tablename__ = "customers"

    id: Mapped[str] = mapped_column(String(16), primary_key=True)
    name: Mapped[str] = mapped_column(String(120))
    email: Mapped[str] = mapped_column(String(254), unique=True)
    tier: Mapped[CustomerTier] = mapped_column(
        _enum(CustomerTier, "customer_tier"),
        default=CustomerTier.STANDARD,
        server_default=CustomerTier.STANDARD.value,
    )
    created_at: Mapped[datetime] = _created_at()

    orders: Mapped[list[Order]] = relationship(back_populates="customer")


class Order(Base):
    __tablename__ = "orders"
    __table_args__ = (
        CheckConstraint("total_paise >= 0", name="total_non_negative"),
        CheckConstraint(
            "refunded_paise >= 0 AND refunded_paise <= total_paise", name="refund_within_total"
        ),
    )

    id: Mapped[str] = mapped_column(String(20), primary_key=True)
    customer_id: Mapped[str] = mapped_column(ForeignKey("customers.id"), index=True)
    status: Mapped[OrderStatus] = mapped_column(_enum(OrderStatus, "order_status"), index=True)
    payment_method: Mapped[PaymentMethod] = mapped_column(_enum(PaymentMethod, "payment_method"))
    placed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    delivered_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    shipping_address: Mapped[dict[str, Any]] = mapped_column(JSON_TYPE)
    currency: Mapped[str] = mapped_column(String(3), default="INR", server_default="INR")
    total_paise: Mapped[int] = mapped_column(BigInteger)
    refunded_paise: Mapped[int] = mapped_column(BigInteger, default=0, server_default="0")

    customer: Mapped[Customer] = relationship(back_populates="orders")
    items: Mapped[list[OrderItem]] = relationship(
        back_populates="order", cascade="all, delete-orphan"
    )
    shipments: Mapped[list[Shipment]] = relationship(
        back_populates="order", cascade="all, delete-orphan"
    )


class OrderItem(Base):
    __tablename__ = "order_items"
    __table_args__ = (
        CheckConstraint("quantity > 0", name="quantity_positive"),
        CheckConstraint("unit_price_paise >= 0", name="price_non_negative"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    order_id: Mapped[str] = mapped_column(ForeignKey("orders.id", ondelete="CASCADE"), index=True)
    sku: Mapped[str] = mapped_column(String(32))
    name: Mapped[str] = mapped_column(String(200))
    category: Mapped[str] = mapped_column(String(40))
    quantity: Mapped[int] = mapped_column(Integer)
    unit_price_paise: Mapped[int] = mapped_column(BigInteger)
    final_sale: Mapped[bool] = mapped_column(Boolean, default=False, server_default=false())

    order: Mapped[Order] = relationship(back_populates="items")


class Shipment(Base):
    __tablename__ = "shipments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    order_id: Mapped[str] = mapped_column(ForeignKey("orders.id", ondelete="CASCADE"), index=True)
    carrier: Mapped[str] = mapped_column(String(40))
    tracking_number: Mapped[str] = mapped_column(String(40), unique=True)
    status: Mapped[ShipmentStatus] = mapped_column(_enum(ShipmentStatus, "shipment_status"))
    shipped_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    estimated_delivery: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    delivered_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    events: Mapped[list[dict[str, Any]]] = mapped_column(JSON_TYPE, default=list)

    order: Mapped[Order] = relationship(back_populates="shipments")


# --------------------------------------------------------------------- support


class Ticket(Base):
    __tablename__ = "tickets"
    __table_args__ = (
        CheckConstraint("tier IS NULL OR (tier >= 0 AND tier <= 3)", name="tier_range"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    customer_id: Mapped[str] = mapped_column(ForeignKey("customers.id"), index=True)
    external_id: Mapped[str | None] = mapped_column(String(64), unique=True)
    channel: Mapped[str] = mapped_column(String(20), default="chat", server_default="chat")
    status: Mapped[TicketStatus] = mapped_column(
        _enum(TicketStatus, "ticket_status"),
        default=TicketStatus.RECEIVED,
        server_default=TicketStatus.RECEIVED.value,
        index=True,
    )
    tier: Mapped[int | None] = mapped_column(Integer)
    intent: Mapped[str | None] = mapped_column(String(40))
    created_at: Mapped[datetime] = _created_at()
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
    closed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    messages: Mapped[list[Message]] = relationship(
        back_populates="ticket", cascade="all, delete-orphan", order_by="Message.created_at"
    )
    runs: Mapped[list[Run]] = relationship(back_populates="ticket", cascade="all, delete-orphan")


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    ticket_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("tickets.id", ondelete="CASCADE"), index=True
    )
    role: Mapped[MessageRole] = mapped_column(_enum(MessageRole, "message_role"))
    content: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = _created_at()

    ticket: Mapped[Ticket] = relationship(back_populates="messages")


class Run(Base):
    """One agent execution for a ticket. Token and cost totals are rolled up from run_steps."""

    __tablename__ = "runs"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    ticket_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("tickets.id", ondelete="CASCADE"), index=True
    )
    status: Mapped[RunStatus] = mapped_column(
        _enum(RunStatus, "run_status"),
        default=RunStatus.RUNNING,
        server_default=RunStatus.RUNNING.value,
    )
    mode: Mapped[RunMode] = mapped_column(_enum(RunMode, "run_mode"))
    model_tier: Mapped[str | None] = mapped_column(String(20))
    prompt_version: Mapped[str | None] = mapped_column(String(40))
    outcome: Mapped[str | None] = mapped_column(String(40))
    tokens_in: Mapped[int] = mapped_column(BigInteger, default=0, server_default="0")
    tokens_out: Mapped[int] = mapped_column(BigInteger, default=0, server_default="0")
    shadow_cost_usd: Mapped[Decimal] = mapped_column(
        Numeric(16, 10), default=Decimal("0"), server_default="0"
    )
    error: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    ticket: Mapped[Ticket] = relationship(back_populates="runs")
    steps: Mapped[list[RunStep]] = relationship(
        back_populates="run", cascade="all, delete-orphan", order_by="RunStep.idx"
    )


class RunStep(Base):
    """One traced step: a pipeline stage, an LLM call, or a tool call."""

    __tablename__ = "run_steps"
    __table_args__ = (UniqueConstraint("run_id", "idx", name="uq_run_steps_run_id_idx"),)

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    run_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"), index=True)
    idx: Mapped[int] = mapped_column(Integer)
    kind: Mapped[StepKind] = mapped_column(_enum(StepKind, "step_kind"))
    name: Mapped[str] = mapped_column(String(80))
    input: Mapped[dict[str, Any] | None] = mapped_column(JSON_TYPE)
    output: Mapped[dict[str, Any] | None] = mapped_column(JSON_TYPE)
    model: Mapped[str | None] = mapped_column(String(80))
    tokens_in: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    tokens_out: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    latency_ms: Mapped[int | None] = mapped_column(Integer)
    shadow_cost_usd: Mapped[Decimal] = mapped_column(
        Numeric(16, 10), default=Decimal("0"), server_default="0"
    )
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = _created_at()

    run: Mapped[Run] = relationship(back_populates="steps")


class Action(Base):
    """A tool action requested by the agent.

    The unique idempotency key stops a retry from executing the same action twice.
    """

    __tablename__ = "actions"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    run_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"), index=True)
    ticket_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("tickets.id", ondelete="CASCADE"), index=True
    )
    tool_name: Mapped[str] = mapped_column(String(60))
    args: Mapped[dict[str, Any]] = mapped_column(JSON_TYPE)
    idempotency_key: Mapped[str] = mapped_column(String(128), unique=True)
    status: Mapped[ActionStatus] = mapped_column(
        _enum(ActionStatus, "action_status"),
        default=ActionStatus.PENDING,
        server_default=ActionStatus.PENDING.value,
    )
    result: Mapped[dict[str, Any] | None] = mapped_column(JSON_TYPE)
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = _created_at()
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class Approval(Base):
    """A request for a human to approve one proposed action.

    ``payload_hash`` fingerprints the exact action (tool, arguments, customer) at the moment it
    was proposed. The action is executed only if its stored arguments still hash to the same
    value, so what a human approved is what runs. A request that is not decided before
    ``expires_at`` can no longer be approved.
    """

    __tablename__ = "approvals"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    action_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("actions.id", ondelete="CASCADE"), unique=True
    )
    ticket_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("tickets.id", ondelete="CASCADE"), index=True
    )
    payload_hash: Mapped[str] = mapped_column(String(64))
    reason: Mapped[str] = mapped_column(String(60))  # why a human is needed: shadow_mode, ...
    status: Mapped[ApprovalStatus] = mapped_column(
        _enum(ApprovalStatus, "approval_status"),
        default=ApprovalStatus.PENDING,
        server_default=ApprovalStatus.PENDING.value,
        index=True,
    )
    requested_at: Mapped[datetime] = _created_at()
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    decided_by: Mapped[str | None] = mapped_column(String(64))
    decided_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    decision_note: Mapped[str | None] = mapped_column(String(300))


class AuditLog(Base):
    """Append-only record of every action. On Postgres a trigger rejects UPDATE/DELETE/TRUNCATE.

    ticket_id and run_id are deliberately plain columns without foreign keys, so audit rows never
    block or cascade with deletes elsewhere.
    """

    __tablename__ = "audit_log"

    id: Mapped[int] = mapped_column(BIGINT_PK, primary_key=True, autoincrement=True)
    ts: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )
    actor_type: Mapped[ActorType] = mapped_column(_enum(ActorType, "actor_type"))
    actor_id: Mapped[str | None] = mapped_column(String(64))
    action: Mapped[str] = mapped_column(String(80))
    ticket_id: Mapped[uuid.UUID | None] = mapped_column(Uuid, index=True)
    run_id: Mapped[uuid.UUID | None] = mapped_column(Uuid)
    target_type: Mapped[str | None] = mapped_column(String(40))
    target_id: Mapped[str | None] = mapped_column(String(64))
    details: Mapped[dict[str, Any] | None] = mapped_column(JSON_TYPE)


class Control(Base):
    """A switch an operator can flip without a deploy: one row per control, read on every use."""

    __tablename__ = "controls"

    name: Mapped[str] = mapped_column(String(40), primary_key=True)
    value: Mapped[str] = mapped_column(String(20))
    reason: Mapped[str | None] = mapped_column(String(300))
    updated_by: Mapped[str] = mapped_column(String(64))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
