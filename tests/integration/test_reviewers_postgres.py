"""Accounts and sessions on a real Postgres (migration 0007, uniqueness, cascades, locking)."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from kartly.reviewers import AuthError, ReviewerError, Reviewers
from kartly.store.models import AdminSession

pytestmark = pytest.mark.postgres
PW = "correct horse battery"


def make(factory: sessionmaker[Session]) -> Reviewers:
    return Reviewers(factory, scrypt_n=2**10)


def test_sign_in_resolve_and_sign_out(pg_factory: sessionmaker[Session]) -> None:
    r = make(pg_factory)
    r.create("maya", PW, "admin")
    login = r.login("maya", PW)
    info = r.resolve(login.token)
    assert info is not None and info.reviewer.username == "maya" and info.csrf == login.csrf
    r.logout(login.token)
    assert r.resolve(login.token) is None


def test_a_disabled_account_loses_its_sessions_and_lockout_is_stored(
    pg_factory: sessionmaker[Session],
) -> None:
    r = make(pg_factory)
    r.create("maya", PW, "admin")
    r.create("omar", PW)
    token = r.login("omar", PW).token
    r.set_active("omar", False)
    assert r.resolve(token) is None
    r.set_active("omar", True)
    for _ in range(5):
        with pytest.raises(AuthError):
            r.login("omar", "nope nope nope")
    assert next(v for v in r.all() if v.username == "omar").locked is True
    r.unlock("omar")
    assert r.login("omar", PW)


def test_two_accounts_with_one_name_cannot_both_be_created(
    pg_factory: sessionmaker[Session],
) -> None:
    r = make(pg_factory)

    def attempt(_: int) -> bool:
        try:
            r.create("maya", PW)
            return True
        except ReviewerError:
            return False

    with ThreadPoolExecutor(4) as pool:
        made = list(pool.map(attempt, range(4)))
    assert made.count(True) == 1
    assert len(r.all()) == 1


def test_two_admins_cannot_disable_each_other_at_the_same_moment(
    pg_factory: sessionmaker[Session],
) -> None:
    r = make(pg_factory)
    r.create("maya", PW, "admin")
    r.create("zoe", PW, "admin")

    def disable(name: str) -> bool:
        try:
            r.set_active(name, False)
            return True
        except ReviewerError:
            return False

    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(disable, ["maya", "zoe"]))
    assert results.count(True) == 1  # one admin always remains
    assert sum(1 for v in r.all() if v.active and v.role == "admin") == 1


def test_purging_removes_only_expired_sessions(pg_factory: sessionmaker[Session]) -> None:
    r = Reviewers(pg_factory, scrypt_n=2**10, session_ttl_s=-1)  # born expired
    r.create("maya", PW)
    r.login("maya", PW)
    assert r.purge_expired() == 1
    with pg_factory() as session:
        assert session.scalar(select(func.count()).select_from(AdminSession)) == 0
