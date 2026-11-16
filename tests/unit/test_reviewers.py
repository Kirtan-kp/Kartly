"""Reviewer accounts, passwords, lockout and sessions."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from kartly.reviewers import (
    AuthError,
    ReviewerError,
    ReviewerNotFoundError,
    Reviewers,
    check_password_rules,
    hash_password,
    main,
    verify_password,
)
from kartly.store.models import ActorType, AdminSession, AuditLog, Reviewer

T0 = datetime(2026, 11, 15, 9, 0, tzinfo=UTC)
GOOD = "correct horse battery"
FAST = 2**10  # scrypt cost: low so the tests are quick; the real default is 2**14


class Clock:
    def __init__(self) -> None:
        self.now = T0

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += timedelta(seconds=seconds)


def service(factory: sessionmaker[Session], clock: Clock | None = None, **kw: Any) -> Reviewers:
    return Reviewers(factory, clock=clock or Clock(), scrypt_n=FAST, **kw)


def audit(factory: sessionmaker[Session], action: str) -> list[AuditLog]:
    with factory() as session:
        return list(session.scalars(select(AuditLog).where(AuditLog.action == action)))


# ------------------------------------------------------------------------------ passwords


def test_a_password_verifies_only_against_its_own_hash() -> None:
    stored = hash_password(GOOD, n=FAST)
    assert stored.startswith("scrypt$1024$8$1$") and GOOD not in stored
    assert verify_password(GOOD, stored) is True
    assert verify_password(GOOD + "x", stored) is False
    assert verify_password("", stored) is False
    assert hash_password(GOOD, n=FAST) != stored  # a new salt every time


@pytest.mark.parametrize(
    "stored",
    [
        "",
        "plain",
        "md5$1$2$3$4$5",
        "scrypt$x$8$1$AAAA$AAAA",
        "scrypt$1024$8$1$!!$!!",
        "scrypt$2$8$1$AAAA$AAAA",
    ],
)
def test_a_malformed_record_never_matches_and_never_raises(stored: str) -> None:
    assert verify_password(GOOD, stored) is False


def test_absurd_cost_parameters_are_refused_not_run() -> None:
    assert verify_password(GOOD, "scrypt$1073741824$8$1$AAAA$AAAA") is False


@pytest.mark.parametrize(
    ("password", "fragment"),
    [
        ("short", "12 to 200"),
        ("x" * 201, "12 to 200"),
        ("maya.reviewer", "too easy"),  # the username itself
        ("aaaaaaaaaaaaaaaa", "too easy"),  # one character repeated
    ],
)
def test_weak_passwords_are_refused(password: str, fragment: str) -> None:
    with pytest.raises(ReviewerError, match=fragment):
        check_password_rules("maya.reviewer", password)
    check_password_rules("maya.reviewer", GOOD)


# ------------------------------------------------------------------------------ accounts


def test_an_account_is_created_with_a_normalised_name_and_audited(
    seeded_factory: sessionmaker[Session],
) -> None:
    r = service(seeded_factory)
    view = r.create("  Maya.R  ", GOOD, "admin", actor="cli")
    assert (view.username, view.role, view.active, view.locked) == ("maya.r", "admin", True, False)
    with seeded_factory() as session:
        row = session.scalar(select(Reviewer))
        assert row is not None and GOOD not in row.password_hash
    [event] = audit(seeded_factory, "reviewer.created")
    assert event.actor_type == ActorType.HUMAN and event.target_id == "maya.r"
    assert event.details == {"role": "admin"}


@pytest.mark.parametrize(
    "name", ["", "ab", "a" * 33, "has space", "Bad!", "-lead", "trail-", "é" * 5]
)
def test_bad_usernames_are_refused(seeded_factory: sessionmaker[Session], name: str) -> None:
    with pytest.raises(ReviewerError, match="username"):
        service(seeded_factory).create(name, GOOD)


def test_roles_and_duplicates_are_checked(seeded_factory: sessionmaker[Session]) -> None:
    r = service(seeded_factory)
    with pytest.raises(ReviewerError, match="role"):
        r.create("maya", GOOD, "root")
    r.create("maya", GOOD)
    with pytest.raises(ReviewerError, match="already exists"):
        r.create("MAYA", GOOD)
    assert [v.username for v in r.all()] == ["maya"]


def test_the_last_active_admin_cannot_be_disabled(seeded_factory: sessionmaker[Session]) -> None:
    r = service(seeded_factory)
    r.create("maya", GOOD, "admin")
    r.create("omar", GOOD, "reviewer")
    with pytest.raises(ReviewerError, match="last active admin"):
        r.set_active("maya", False)
    r.set_active("omar", False)  # a reviewer is fine
    r.create("zoe", GOOD, "admin")
    r.set_active("maya", False)  # not the last any more
    with pytest.raises(ReviewerError, match="last active admin"):
        r.set_active("zoe", False)
    r.set_active("maya", True)
    assert {v.username: v.active for v in r.all()} == {"maya": True, "omar": False, "zoe": True}
    assert [a.action for a in audit(seeded_factory, "reviewer.deactivated")] == [
        "reviewer.deactivated"
    ] * 2


def test_unknown_accounts_are_reported_as_missing(seeded_factory: sessionmaker[Session]) -> None:
    r = service(seeded_factory)
    with pytest.raises(ReviewerNotFoundError):
        r.set_active("ghost", False)
    with pytest.raises(ReviewerNotFoundError):
        r.set_password("ghost", GOOD)
    with pytest.raises(ReviewerNotFoundError):
        r.unlock("ghost")


# ------------------------------------------------------------------------------ signing in


def test_sign_in_gives_a_session_whose_cookie_is_not_stored(
    seeded_factory: sessionmaker[Session],
) -> None:
    r = service(seeded_factory)
    r.create("maya", GOOD, "admin")
    result = r.login("MAYA ", GOOD)
    assert result.reviewer.username == "maya" and result.reviewer.last_login_at == T0
    assert len(result.token) >= 40 and result.csrf and result.token != result.csrf
    assert result.expires_at == T0 + timedelta(hours=12)
    with seeded_factory() as session:
        [row] = session.scalars(select(AdminSession)).all()
        assert result.token not in (row.token_hash, row.csrf) and len(row.token_hash) == 64
    info = r.resolve(result.token)
    assert info is not None and info.reviewer.username == "maya" and info.csrf == result.csrf
    assert [e.actor_id for e in audit(seeded_factory, "admin.login")] == ["maya"]


def test_every_failure_looks_the_same_and_still_pays_for_a_hash(
    seeded_factory: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
) -> None:
    r = service(seeded_factory)
    r.create("maya", GOOD)
    r.create("omar", GOOD)
    r.set_active("omar", False)
    calls: list[str] = []
    real = verify_password

    def spy(password: str, stored: str) -> bool:
        calls.append(password)
        return real(password, stored)

    monkeypatch.setattr("kartly.reviewers.verify_password", spy)
    attempts = [("maya", "wrong password!"), ("nobody", GOOD), ("omar", GOOD)]
    for name, password in attempts:
        with pytest.raises(AuthError) as caught:
            r.login(name, password)
        assert str(caught.value) == "invalid username or password"
    assert len(calls) == 3  # the unknown name and the disabled account did the same work
    reasons = [
        e.details["reason"] for e in audit(seeded_factory, "admin.login_failed") if e.details
    ]
    assert reasons == ["bad_password", "unknown_user", "inactive"]


def test_five_wrong_passwords_lock_the_account_for_fifteen_minutes(
    seeded_factory: sessionmaker[Session],
) -> None:
    clock = Clock()
    r = service(seeded_factory, clock)
    r.create("maya", GOOD)
    for _ in range(4):
        with pytest.raises(AuthError):
            r.login("maya", "nope nope nope")
    assert r.all()[0].locked is False
    with pytest.raises(AuthError):
        r.login("maya", "nope nope nope")  # the fifth
    assert r.all()[0].locked is True
    with pytest.raises(AuthError):
        r.login("maya", GOOD)  # even the right password
    clock.advance(899)
    with pytest.raises(AuthError):
        r.login("maya", GOOD)
    clock.advance(2)
    assert r.login("maya", GOOD).reviewer.locked is False
    reasons = [
        e.details["reason"] for e in audit(seeded_factory, "admin.login_failed") if e.details
    ]
    assert reasons.count("bad_password_now_locked") == 1 and reasons.count("locked") == 2


def test_a_good_sign_in_resets_the_failure_count(seeded_factory: sessionmaker[Session]) -> None:
    r = service(seeded_factory)
    r.create("maya", GOOD)
    for _ in range(4):
        with pytest.raises(AuthError):
            r.login("maya", "nope nope nope")
    r.login("maya", GOOD)
    for _ in range(4):  # four more would lock it if the count had not been reset
        with pytest.raises(AuthError):
            r.login("maya", "nope nope nope")
    assert r.login("maya", GOOD)


def test_attempts_while_locked_do_not_extend_the_lock(
    seeded_factory: sessionmaker[Session],
) -> None:
    clock = Clock()
    r = service(seeded_factory, clock, max_failures=2, lockout_s=100)
    r.create("maya", GOOD)
    for _ in range(2):
        with pytest.raises(AuthError):
            r.login("maya", "nope nope nope")
    clock.advance(60)
    for _ in range(5):
        with pytest.raises(AuthError):
            r.login("maya", "nope nope nope")  # a stranger hammering a locked account
    clock.advance(41)  # 101 s after the lock began
    assert r.login("maya", GOOD)


def test_unlock_lifts_a_lock_at_once(seeded_factory: sessionmaker[Session]) -> None:
    r = service(seeded_factory, max_failures=1)
    r.create("maya", GOOD)
    with pytest.raises(AuthError):
        r.login("maya", "nope nope nope")
    with pytest.raises(AuthError):
        r.login("maya", GOOD)
    r.unlock("maya", actor="omar")
    assert r.login("maya", GOOD)
    assert audit(seeded_factory, "reviewer.unlocked")[0].actor_id == "omar"


def test_a_very_long_password_cannot_be_used_to_burn_time(
    seeded_factory: sessionmaker[Session],
) -> None:
    r = service(seeded_factory)
    r.create("maya", GOOD)
    with pytest.raises(AuthError):
        r.login("maya", "x" * 1_000_000)


# ------------------------------------------------------------------------------ sessions


def test_a_session_ends_after_the_absolute_lifetime_even_if_used(
    seeded_factory: sessionmaker[Session],
) -> None:
    clock = Clock()
    r = service(seeded_factory, clock, session_ttl_s=3600, session_idle_s=1800)
    r.create("maya", GOOD)
    token = r.login("maya", GOOD).token
    for _ in range(5):  # used every 10 minutes: never idle
        clock.advance(600)
        assert r.resolve(token) is not None
    clock.advance(600)  # 60 minutes in: the lifetime is over
    assert r.resolve(token) is None


def test_a_session_ends_when_left_idle(seeded_factory: sessionmaker[Session]) -> None:
    clock = Clock()
    r = service(seeded_factory, clock, session_ttl_s=86400, session_idle_s=1800)
    r.create("maya", GOOD)
    token = r.login("maya", GOOD).token
    clock.advance(1799)
    assert r.resolve(token) is not None  # this use keeps it alive
    clock.advance(1799)
    assert r.resolve(token) is not None
    clock.advance(1801)
    assert r.resolve(token) is None
    with seeded_factory() as session:
        assert session.scalars(select(AdminSession)).all() == []  # and it was removed


def test_use_is_recorded_at_most_once_a_minute(seeded_factory: sessionmaker[Session]) -> None:
    clock = Clock()
    r = service(seeded_factory, clock)
    r.create("maya", GOOD)
    token = r.login("maya", GOOD).token

    def seen() -> datetime:
        with seeded_factory() as session:
            row = session.scalars(select(AdminSession)).one()
            return row.last_seen_at.replace(tzinfo=UTC)

    clock.advance(30)
    r.resolve(token)
    assert seen() == T0  # too soon to bother the database
    clock.advance(31)
    r.resolve(token)
    assert seen() == T0 + timedelta(seconds=61)


@pytest.mark.parametrize("token", [None, "", "nope", "x" * 500])
def test_unknown_cookies_resolve_to_nobody(
    seeded_factory: sessionmaker[Session], token: str | None
) -> None:
    assert service(seeded_factory).resolve(token) is None


def test_disabling_an_account_ends_its_sessions_at_once(
    seeded_factory: sessionmaker[Session],
) -> None:
    r = service(seeded_factory)
    r.create("maya", GOOD, "admin")
    r.create("omar", GOOD)
    mine, theirs = r.login("maya", GOOD).token, r.login("omar", GOOD).token
    r.set_active("omar", False)
    assert r.resolve(theirs) is None and r.resolve(mine) is not None


def test_changing_a_password_signs_the_person_out_everywhere_and_unlocks(
    seeded_factory: sessionmaker[Session],
) -> None:
    r = service(seeded_factory, max_failures=1)
    r.create("maya", GOOD)
    a, b = r.login("maya", GOOD).token, r.login("maya", GOOD).token
    with pytest.raises(AuthError):
        r.login("maya", "nope nope nope")  # locks it
    r.set_password("maya", "a different passphrase", actor="omar")
    assert r.resolve(a) is None and r.resolve(b) is None
    assert r.login("maya", "a different passphrase")
    with pytest.raises(AuthError):
        r.login("maya", GOOD)  # the old one no longer works
    with pytest.raises(ReviewerError):
        r.set_password("maya", "short")


def test_logout_ends_the_session_and_is_audited_once(
    seeded_factory: sessionmaker[Session],
) -> None:
    r = service(seeded_factory)
    r.create("maya", GOOD)
    token = r.login("maya", GOOD).token
    r.logout(token)
    r.logout(token)
    r.logout(None)
    assert r.resolve(token) is None and len(audit(seeded_factory, "admin.logout")) == 1


def test_expired_sessions_can_be_purged(seeded_factory: sessionmaker[Session]) -> None:
    clock = Clock()
    r = service(seeded_factory, clock, session_ttl_s=100, session_idle_s=1000)
    r.create("maya", GOOD)
    r.login("maya", GOOD)
    clock.advance(50)
    r.login("maya", GOOD)
    clock.advance(60)  # the first is over its lifetime, the second is not
    assert r.purge_expired() == 1
    assert r.purge_expired() == 0


# ------------------------------------------------------------------------------ command line


@pytest.fixture
def cli(seeded_factory: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("kartly.store.db.make_engine", lambda *a, **k: seeded_factory.kw["bind"])
    monkeypatch.setattr("sqlalchemy.engine.Engine.dispose", lambda self, *a, **k: None)
    monkeypatch.setattr("kartly.config.load_env", lambda: None)


def test_the_command_line_manages_accounts(
    cli: None,
    seeded_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr("getpass.getpass", lambda prompt="": GOOD)
    assert main(["add", "maya", "--role", "admin"]) == 0
    assert main(["add", "omar"]) == 0
    assert main(["add", "omar"]) == 2  # exists
    capsys.readouterr()
    assert main(["list"]) == 0
    listing = capsys.readouterr().out
    assert "maya" in listing and "admin" in listing and "omar" in listing
    assert main(["disable", "omar"]) == 0 and main(["list"]) == 0
    assert "disabled" in capsys.readouterr().out
    assert main(["disable", "maya"]) == 2  # the last admin
    assert main(["enable", "omar"]) == 0 and main(["unlock", "omar"]) == 0
    assert main(["passwd", "omar"]) == 0
    assert main(["unlock", "ghost"]) == 2


def test_the_command_line_refuses_mismatched_passwords(
    cli: None, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    answers = iter([GOOD, GOOD + "!"])
    monkeypatch.setattr("getpass.getpass", lambda prompt="": next(answers))
    assert main(["add", "maya"]) == 2
    assert "differ" in capsys.readouterr().out


def test_after_a_lock_ends_one_typo_does_not_lock_again(
    seeded_factory: sessionmaker[Session],
) -> None:
    clock = Clock()
    r = service(seeded_factory, clock, max_failures=3, lockout_s=60)
    r.create("maya", GOOD)
    for _ in range(3):
        with pytest.raises(AuthError):
            r.login("maya", "nope nope nope")
    clock.advance(61)
    with pytest.raises(AuthError):
        r.login("maya", "nope nope nope")  # one typo after the lock ended
    assert r.all()[0].locked is False
    assert r.login("maya", GOOD)
