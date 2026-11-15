"""Alert state and the whole check on a real Postgres."""

from __future__ import annotations

import pytest
from sqlalchemy.orm import Session, sessionmaker

from kartly.alerts import AlertManager
from kartly.controls import Controls
from kartly.store.models import Job, JobStatus, Ticket

pytestmark = pytest.mark.postgres


def test_a_check_stores_state_and_announces_once(pg_factory: sessionmaker[Session]) -> None:
    with pg_factory() as session:
        ticket = Ticket(customer_id="C-0001")
        session.add(ticket)
        session.flush()
        session.add(Job(ticket_id=ticket.id, status=JobStatus.DEAD, available_at=ticket.created_at))
        session.commit()
    sent: list[str] = []
    manager = AlertManager(
        pg_factory, controls=Controls(pg_factory), notifier=lambda a: sent.append(a.rule)
    )
    assert [f.rule for f in manager.check().firing] == ["dead_letters"]
    assert manager.check().fired == []
    assert sent == ["dead_letters"]
    [state] = manager.states()
    assert state.status == "firing" and state.details == {"dead": 1}
