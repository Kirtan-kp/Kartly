"""Reviewer accounts, passwords and browser sessions for the admin interface (ADR-056).

    python -m kartly.reviewers add maya --role admin     # asks for a password
    python -m kartly.reviewers list | passwd | disable | enable | unlock NAME

What it guarantees:

- Passwords are stored as salted scrypt hashes and compared in constant time. Unknown users cost
  the same time as wrong passwords, and every failure looks alike to the caller.
- Five wrong passwords lock the account for 15 minutes (a locked account says nothing different).
  ``unlock`` from the command line lifts it. A locked, disabled or unknown account never signs in
  even with the right password.
- A session is a random 256-bit cookie value; only its SHA-256 is stored. It ends after 12 hours
  at most and after 2 hours without use. Changing a password or disabling an account ends every
  session of that person. Each session has its own CSRF token.
- The last active admin cannot be disabled, so nobody can lock everyone out.
- Sign-ins, failures, logouts and account changes go to the audit log.
"""

from __future__ import annotations

import argparse
import base64
import getpass
import hashlib
import hmac
import re
import secrets
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import delete, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from kartly.sanitize import clean_str
from kartly.store.models import ActorType, AdminSession, AuditLog, Reviewer
from kartly.tools.orders import utc

ROLES = ("reviewer", "admin")
USERNAME = re.compile(r"[a-z0-9][a-z0-9_.-]{1,30}[a-z0-9]")  # 3 to 32 characters
MIN_PASSWORD = 12
MAX_PASSWORD = 200
SCRYPT_N, SCRYPT_R, SCRYPT_P = 2**14, 8, 1
SESSION_TTL_S = 12 * 3600.0
SESSION_IDLE_S = 2 * 3600.0
MAX_FAILURES = 5
LOCKOUT_S = 900.0
TOUCH_EVERY_S = 60.0


class AuthError(Exception):
    """Sign-in failed. The message never says why."""


class ReviewerError(ValueError):
    """A request about accounts that cannot be done (a bad name, a weak password, ...)."""


class ReviewerNotFoundError(ReviewerError):
    pass


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


def hash_password(password: str, *, n: int = SCRYPT_N) -> str:
    salt = secrets.token_bytes(16)
    key = hashlib.scrypt(password.encode(), salt=salt, n=n, r=SCRYPT_R, p=SCRYPT_P, dklen=32)
    return f"scrypt${n}${SCRYPT_R}${SCRYPT_P}${_b64(salt)}${_b64(key)}"


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, n, r, p, salt, key = stored.split("$")
        if scheme != "scrypt":
            return False
        expected = base64.b64decode(key)
        actual = hashlib.scrypt(
            password.encode(),
            salt=base64.b64decode(salt),
            n=int(n),
            r=int(r),
            p=int(p),
            dklen=len(expected),
            maxmem=256 * 1024 * 1024,
        )
    except (ValueError, TypeError):  # a malformed record or absurd parameters: never a match
        return False
    return hmac.compare_digest(actual, expected)


def check_password_rules(username: str, password: str) -> None:
    if not MIN_PASSWORD <= len(password) <= MAX_PASSWORD:
        raise ReviewerError(f"the password must be {MIN_PASSWORD} to {MAX_PASSWORD} characters")
    if password.lower() == username.lower() or len(set(password)) < 5:
        raise ReviewerError("that password is too easy to guess")


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


@dataclass(frozen=True)
class ReviewerView:
    id: uuid.UUID
    username: str
    role: str
    active: bool
    locked: bool
    created_at: datetime
    last_login_at: datetime | None


@dataclass(frozen=True)
class Login:
    token: str  # the cookie value; shown once
    csrf: str
    reviewer: ReviewerView
    expires_at: datetime


@dataclass(frozen=True)
class SessionInfo:
    reviewer: ReviewerView
    csrf: str


class Reviewers:
    def __init__(
        self,
        session_factory: Callable[[], Session] | sessionmaker[Session],
        *,
        clock: Callable[[], datetime] | None = None,
        session_ttl_s: float = SESSION_TTL_S,
        session_idle_s: float = SESSION_IDLE_S,
        max_failures: int = MAX_FAILURES,
        lockout_s: float = LOCKOUT_S,
        scrypt_n: int = SCRYPT_N,
    ) -> None:
        self._factory = session_factory
        self._clock = clock or (lambda: datetime.now(UTC))
        self._ttl = timedelta(seconds=session_ttl_s)
        self._idle = timedelta(seconds=session_idle_s)
        self._max_failures = max_failures
        self._lockout = timedelta(seconds=lockout_s)
        self._n = scrypt_n
        self._dummy = hash_password("not-a-real-password", n=scrypt_n)  # for unknown names

    # ------------------------------------------------------------------ accounts

    def _view(self, r: Reviewer, now: datetime) -> ReviewerView:
        return ReviewerView(
            r.id,
            r.username,
            r.role,
            r.active,
            r.locked_until is not None and utc(r.locked_until) > now,
            utc(r.created_at),
            utc(r.last_login_at) if r.last_login_at else None,
        )

    @staticmethod
    def _norm(username: str) -> str:
        return username.strip().lower()

    @staticmethod
    def _audit(
        session: Session,
        action: str,
        actor: str,
        target: str,
        now: datetime,
        details: dict[str, Any] | None = None,
        *,
        human: bool = True,
    ) -> None:
        session.add(
            AuditLog(
                actor_type=ActorType.HUMAN if human else ActorType.SYSTEM,
                actor_id=clean_str(actor, 64),
                action=action,
                target_type="reviewer",
                target_id=clean_str(target, 64),
                details=details,
                ts=now,
            )
        )

    def create(
        self, username: str, password: str, role: str = "reviewer", *, actor: str = "cli"
    ) -> ReviewerView:
        name = self._norm(username)
        if not USERNAME.fullmatch(name):
            raise ReviewerError(
                "the username must be 3 to 32 characters: lowercase letters, digits, . _ -"
            )
        if role not in ROLES:
            raise ReviewerError(f"the role must be one of {', '.join(ROLES)}")
        check_password_rules(name, password)
        now = self._clock()
        with self._factory() as session:
            if session.scalar(select(Reviewer.id).where(Reviewer.username == name)) is not None:
                raise ReviewerError(f"{name!r} already exists")
            row = Reviewer(
                username=name, password_hash=hash_password(password, n=self._n), role=role
            )
            row.created_at = now
            session.add(row)
            self._audit(session, "reviewer.created", actor, name, now, {"role": role})
            try:
                session.commit()
            except IntegrityError:  # someone created the same name between our check and commit
                session.rollback()
                raise ReviewerError(f"{name!r} already exists") from None
            return self._view(row, now)

    def _get(self, session: Session, username: str, *, lock: bool = False) -> Reviewer:
        query = select(Reviewer).where(Reviewer.username == self._norm(username))
        row = session.scalar(query.with_for_update() if lock else query)
        if row is None:
            raise ReviewerNotFoundError(f"no reviewer {self._norm(username)!r}")
        return row

    def set_password(self, username: str, password: str, *, actor: str = "cli") -> None:
        now = self._clock()
        with self._factory() as session:
            row = self._get(session, username, lock=True)
            check_password_rules(row.username, password)
            row.password_hash = hash_password(password, n=self._n)
            row.failed_attempts, row.locked_until = 0, None
            session.execute(delete(AdminSession).where(AdminSession.reviewer_id == row.id))
            self._audit(session, "reviewer.password_changed", actor, row.username, now)
            session.commit()

    def set_active(self, username: str, active: bool, *, actor: str = "cli") -> None:
        now = self._clock()
        with self._factory() as session:
            row = self._get(session, username)
            if not active and row.role == "admin":
                # Two admins disabling each other must queue up: lock every admin row, always in
                # the same order (so they cannot deadlock), then look at the committed state.
                session.execute(
                    select(Reviewer.id)
                    .where(Reviewer.role == "admin")
                    .order_by(Reviewer.id)
                    .with_for_update()
                ).all()
                session.refresh(row)
            else:
                row = self._get(session, username, lock=True)
            if not active and row.active and row.role == "admin":
                admins = session.scalar(
                    select(func.count())
                    .select_from(Reviewer)
                    .where(Reviewer.role == "admin", Reviewer.active.is_(True))
                )
                if (admins or 0) <= 1:
                    raise ReviewerError("that is the last active admin; add another one first")
            row.active = active
            if not active:
                session.execute(delete(AdminSession).where(AdminSession.reviewer_id == row.id))
            self._audit(
                session,
                "reviewer.activated" if active else "reviewer.deactivated",
                actor,
                row.username,
                now,
            )
            session.commit()

    def unlock(self, username: str, *, actor: str = "cli") -> None:
        now = self._clock()
        with self._factory() as session:
            row = self._get(session, username, lock=True)
            row.failed_attempts, row.locked_until = 0, None
            self._audit(session, "reviewer.unlocked", actor, row.username, now)
            session.commit()

    def all(self) -> list[ReviewerView]:
        now = self._clock()
        with self._factory() as session:
            return [
                self._view(r, now)
                for r in session.scalars(select(Reviewer).order_by(Reviewer.username))
            ]

    # ------------------------------------------------------------------ signing in

    def login(self, username: str, password: str) -> Login:
        """A session for a valid username and password. Any failure raises AuthError."""
        name = self._norm(username)[:64]
        now = self._clock()
        with self._factory() as session:
            row = session.scalar(
                select(Reviewer).where(Reviewer.username == name).with_for_update()
            )
            # Always do the expensive hash, whether or not the account exists or may sign in.
            matches = verify_password(
                password[:MAX_PASSWORD], row.password_hash if row else self._dummy
            )
            reason = None
            if row is None:
                reason = "unknown_user"
            elif row.locked_until is not None and utc(row.locked_until) > now:
                reason = "locked"
            elif not row.active:
                reason = "inactive"
            elif not matches:
                reason = "bad_password"
            if reason is not None or row is None:
                if row is not None and reason == "bad_password":
                    row.failed_attempts += 1
                    if row.failed_attempts >= self._max_failures:
                        row.locked_until = now + self._lockout
                        row.failed_attempts = 0
                        reason = "bad_password_now_locked"
                self._audit(
                    session,
                    "admin.login_failed",
                    row.username if row else "login",
                    clean_str(name, 40),
                    now,
                    {"reason": reason},
                    human=row is not None,
                )
                session.commit()
                raise AuthError("invalid username or password")
            row.failed_attempts, row.locked_until, row.last_login_at = 0, None, now
            token = secrets.token_urlsafe(32)
            csrf = secrets.token_urlsafe(24)
            session.add(
                AdminSession(
                    token_hash=_token_hash(token),
                    reviewer_id=row.id,
                    csrf=csrf,
                    created_at=now,
                    expires_at=now + self._ttl,
                    last_seen_at=now,
                )
            )
            self._audit(session, "admin.login", row.username, row.username, now)
            session.commit()
            return Login(token, csrf, self._view(row, now), now + self._ttl)

    def resolve(self, token: str | None) -> SessionInfo | None:
        """Who a cookie value belongs to, or None (unknown, expired, idle, or account disabled)."""
        if not token or len(token) > 200:
            return None
        now = self._clock()
        with self._factory() as session:
            found = session.get(AdminSession, _token_hash(token))
            if found is None:
                return None
            reviewer = session.get(Reviewer, found.reviewer_id)
            if (
                reviewer is None
                or not reviewer.active
                or utc(found.expires_at) <= now
                or utc(found.last_seen_at) + self._idle <= now
            ):
                session.delete(found)
                session.commit()
                return None
            if now - utc(found.last_seen_at) > timedelta(seconds=TOUCH_EVERY_S):
                found.last_seen_at = now
                session.commit()
            return SessionInfo(self._view(reviewer, now), found.csrf)

    def logout(self, token: str | None) -> None:
        if not token:
            return
        now = self._clock()
        with self._factory() as session:
            found = session.get(AdminSession, _token_hash(token))
            if found is None:
                return
            reviewer = session.get(Reviewer, found.reviewer_id)
            session.delete(found)
            if reviewer is not None:
                self._audit(session, "admin.logout", reviewer.username, reviewer.username, now)
            session.commit()

    def purge_expired(self) -> int:
        now = self._clock()
        with self._factory() as session:
            result = session.execute(
                delete(AdminSession).where(
                    (AdminSession.expires_at <= now)
                    | (AdminSession.last_seen_at <= now - self._idle)
                )
            )
            session.commit()
            return int(result.rowcount or 0)  # type: ignore[attr-defined]


# ------------------------------------------------------------------------------ command line


def main(argv: Sequence[str] | None = None) -> int:
    from kartly.config import load_env
    from kartly.store.db import make_engine, make_session_factory

    parser = argparse.ArgumentParser(description="Manage admin-interface accounts.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    add = sub.add_parser("add", help="create an account (asks for a password)")
    add.add_argument("username")
    add.add_argument("--role", choices=ROLES, default="reviewer")
    for name in ("passwd", "disable", "enable", "unlock"):
        sub.add_parser(name).add_argument("username")
    sub.add_parser("list")
    args = parser.parse_args(argv)

    load_env()
    engine = make_engine()
    try:
        reviewers = Reviewers(make_session_factory(engine))
        try:
            if args.cmd == "list":
                for r in reviewers.all():
                    state = "locked" if r.locked else ("active" if r.active else "disabled")
                    print(f"{r.username:<20} {r.role:<9} {state}")
            elif args.cmd in ("add", "passwd"):
                password = getpass.getpass("password: ")
                if password != getpass.getpass("again: "):
                    print("error: the passwords differ")
                    return 2
                if args.cmd == "add":
                    reviewers.create(args.username, password, args.role)
                    print(f"created {args.username.strip().lower()} ({args.role})")
                else:
                    reviewers.set_password(args.username, password)
                    print("password changed; the account was signed out everywhere")
            elif args.cmd == "disable":
                reviewers.set_active(args.username, False)
                print("disabled")
            elif args.cmd == "enable":
                reviewers.set_active(args.username, True)
                print("enabled")
            else:
                reviewers.unlock(args.username)
                print("unlocked")
        except ReviewerError as exc:
            print(f"error: {exc}")
            return 2
    finally:
        engine.dispose()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
