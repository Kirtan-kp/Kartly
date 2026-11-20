"""Browser sign-in, CSRF, roles, account management and the ticket list."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session, sessionmaker

from kartly.api import ApiState, create_app
from kartly.api.app import SESSION_COOKIE
from kartly.controls import Controls
from kartly.orchestrator import TicketRunner
from kartly.reviewers import Reviewers
from kartly.store.models import Ticket, TicketStatus
from tests.llm_support import ScriptedClient
from tests.unit.test_api_approvals import (
    AUTH,
    KEY,
    Inline,
    make_gateway,
    open_request,
    pending_id,
    refund_script,
)

T0 = datetime(2026, 11, 15, 9, 0, tzinfo=UTC)
PW = "correct horse battery"
FAST = 2**10


class Clock:
    def __init__(self) -> None:
        self.now = T0

    def __call__(self) -> datetime:
        return self.now


def build(
    factory: sessionmaker[Session],
    script: list[Any] | None = None,
    *,
    clock: Clock | None = None,
    secure: bool = False,
    with_reviewers: bool = True,
) -> TestClient:
    gateway = make_gateway(factory)
    runner = TicketRunner(
        factory, ScriptedClient(script or refund_script()), gateway, model_tier="scripted"
    )
    state = ApiState(
        session_factory=factory,
        runner=runner,
        executor=Inline(),
        api_key=KEY,
        gateway=gateway,
        controls=Controls(factory),
        reviewers=Reviewers(factory, clock=clock or Clock(), scrypt_n=FAST)
        if with_reviewers
        else None,
        cookie_secure=secure,
    )
    return TestClient(create_app(state))


def accounts(http: TestClient) -> Reviewers:
    reviewers = http.app.state.kartly.reviewers  # type: ignore[attr-defined]
    assert isinstance(reviewers, Reviewers)
    return reviewers


def sign_in(http: TestClient, name: str, role: str = "reviewer") -> dict[str, str]:
    """Create the account if needed, sign in, return the headers a browser page would send."""
    reviewers = accounts(http)
    if not any(v.username == name for v in reviewers.all()):
        reviewers.create(name, PW, role)
    got = http.post("/v1/admin/login", json={"username": name, "password": PW})
    assert got.status_code == 200, got.text
    return {"X-CSRF-Token": got.json()["csrf"]}


@pytest.fixture
def http(seeded_factory: sessionmaker[Session]) -> TestClient:
    return build(seeded_factory)


# ------------------------------------------------------------------------------ login


def test_login_sets_a_locked_down_cookie_and_returns_the_csrf_token(http: TestClient) -> None:
    accounts(http).create("maya", PW, "admin")
    got = http.post("/v1/admin/login", json={"username": "Maya", "password": PW})
    assert got.status_code == 200
    body = got.json()
    assert body["username"] == "maya" and body["role"] == "admin" and body["csrf"]
    cookie = got.headers["set-cookie"].lower()
    assert f"{SESSION_COOKIE}=" in cookie and "httponly" in cookie
    assert "samesite=strict" in cookie and "path=/" in cookie and "secure" not in cookie
    assert got.headers["cache-control"] == "no-store"
    assert body["csrf"] not in cookie  # the token is not the cookie


def test_the_cookie_is_marked_secure_when_configured(seeded_factory: sessionmaker[Session]) -> None:
    secure = build(seeded_factory, secure=True)
    accounts(secure).create("maya", PW)
    got = secure.post("/v1/admin/login", json={"username": "maya", "password": PW})
    assert "secure" in got.headers["set-cookie"].lower()


def test_bad_credentials_get_the_same_401_whatever_is_wrong(http: TestClient) -> None:
    accounts(http).create("maya", PW)
    answers = {
        http.post("/v1/admin/login", json={"username": u, "password": p}).text
        for u, p in [("maya", "wrong password!"), ("nobody", PW)]
    }
    assert len(answers) == 1
    got = http.post("/v1/admin/login", json={"username": "maya", "password": "wrong password!"})
    assert got.status_code == 401 and SESSION_COOKIE not in got.headers.get("set-cookie", "")


def test_login_without_accounts_configured_says_so(seeded_factory: sessionmaker[Session]) -> None:
    bare = build(seeded_factory, with_reviewers=False)
    got = bare.post("/v1/admin/login", json={"username": "maya", "password": PW})
    assert got.status_code == 503
    assert bare.get("/v1/approvals").status_code == 401  # and no cookie works either
    assert bare.get("/v1/approvals", headers=AUTH).status_code == 200


def test_login_rejects_unexpected_fields(http: TestClient) -> None:
    got = http.post("/v1/admin/login", json={"username": "a", "password": "b", "role": "admin"})
    assert got.status_code == 422


def test_me_reports_the_session_and_refuses_a_bare_key(http: TestClient) -> None:
    assert http.get("/v1/admin/me").status_code == 401
    assert http.get("/v1/admin/me", headers=AUTH).status_code == 403
    csrf = sign_in(http, "omar")["X-CSRF-Token"]
    got = http.get("/v1/admin/me")
    assert got.json() == {"username": "omar", "role": "reviewer", "csrf": csrf}


def test_five_bad_passwords_lock_the_account_through_the_api(http: TestClient) -> None:
    accounts(http).create("maya", PW)
    for _ in range(5):
        http.post("/v1/admin/login", json={"username": "maya", "password": "wrong password!"})
    assert (
        http.post("/v1/admin/login", json={"username": "maya", "password": PW}).status_code == 401
    )


# ------------------------------------------------------------------------------ sessions


def test_a_session_can_read_without_the_api_key(http: TestClient) -> None:
    assert http.get("/v1/approvals").status_code == 401
    sign_in(http, "omar")
    for path in ("/v1/approvals", "/v1/controls", "/v1/tickets", "/v1/metrics"):
        assert http.get(path).status_code == 200, path


def test_logout_ends_the_session_and_clears_the_cookie(http: TestClient) -> None:
    sign_in(http, "omar")
    token = http.cookies.get(SESSION_COOKIE)
    assert token
    out = http.post("/v1/admin/logout")
    assert out.status_code == 204 and f"{SESSION_COOKIE}=" in out.headers["set-cookie"]
    http.cookies.set(SESSION_COOKIE, token)  # a copy of the old cookie is now worthless
    assert http.get("/v1/approvals").status_code == 401
    assert http.post("/v1/admin/logout").status_code == 204  # and logging out twice is fine


def test_sessions_expire_when_idle_and_at_the_end_of_their_lifetime(
    seeded_factory: sessionmaker[Session],
) -> None:
    clock = Clock()
    web = build(seeded_factory, clock=clock)
    sign_in(web, "omar")
    clock.now += timedelta(hours=1, minutes=59)
    assert web.get("/v1/approvals").status_code == 200
    clock.now += timedelta(hours=2, minutes=1)
    assert web.get("/v1/approvals").status_code == 401  # idle
    sign_in(web, "omar")
    for _ in range(7):  # busy all day, still cut off after 12 hours
        clock.now += timedelta(hours=1, minutes=45)
        web.get("/v1/approvals")
    assert web.get("/v1/approvals").status_code == 401


def test_a_wrong_api_key_is_not_rescued_by_a_valid_cookie(http: TestClient) -> None:
    sign_in(http, "omar")
    assert http.get("/v1/approvals", headers={"X-API-Key": "wrong-key-123456"}).status_code == 401
    assert http.get("/v1/approvals", headers={"X-API-Key": ""}).status_code == 401


def test_a_disabled_account_is_signed_out_immediately(http: TestClient) -> None:
    accounts(http).create("root", PW, "admin")
    sign_in(http, "omar")
    assert http.get("/v1/approvals").status_code == 200
    accounts(http).set_active("omar", False)
    assert http.get("/v1/approvals").status_code == 401


# ------------------------------------------------------------------------------ csrf and actors


def test_changes_from_a_session_need_the_csrf_header(
    seeded_factory: sessionmaker[Session],
) -> None:
    web = build(seeded_factory)
    open_request(web)
    headers = sign_in(web, "omar")
    approval = pending_id_with_cookie(web)
    url = f"/v1/approvals/{approval}/reject"
    assert web.post(url, json={}).status_code == 403
    assert web.post(url, json={}, headers={"X-CSRF-Token": "wrong"}).status_code == 403
    assert web.post(url, json={}, headers={"X-CSRF-Token": ""}).status_code == 403
    assert web.post(url, json={}, headers=headers).status_code == 200


def pending_id_with_cookie(web: TestClient) -> str:
    [only] = web.get("/v1/approvals").json()["approvals"]
    return str(only["id"])


def test_a_session_decides_as_itself_whatever_the_body_says(
    seeded_factory: sessionmaker[Session],
) -> None:
    web = build(seeded_factory)
    open_request(web)
    headers = sign_in(web, "omar")
    approval = pending_id_with_cookie(web)
    got = web.post(
        f"/v1/approvals/{approval}/approve", json={"reviewer": "somebody-else"}, headers=headers
    )
    assert got.status_code == 200, got.text
    listed = web.get("/v1/approvals?status=approved").json()["approvals"]
    assert [a["decided_by"] for a in listed] == ["human:omar"]


def test_the_api_key_must_still_say_who_is_deciding(
    seeded_factory: sessionmaker[Session],
) -> None:
    web = build(seeded_factory)
    open_request(web)
    approval = pending_id(web)
    assert web.post(f"/v1/approvals/{approval}/approve", json={}, headers=AUTH).status_code == 422
    ok = web.post(f"/v1/approvals/{approval}/approve", json={"reviewer": "maya"}, headers=AUTH)
    assert ok.status_code == 200


def test_a_cookie_session_needs_no_csrf_for_reading_only(http: TestClient) -> None:
    sign_in(http, "omar")
    assert http.get("/v1/controls").status_code == 200
    assert http.put("/v1/controls/agent", json={"value": "off"}).status_code == 403


# ------------------------------------------------------------------------------ roles


def test_a_reviewer_can_stop_things_but_not_start_them(http: TestClient) -> None:
    headers = sign_in(http, "omar")
    assert (
        http.put("/v1/controls/writes", json={"value": "frozen"}, headers=headers).status_code
        == 200
    )
    assert http.put("/v1/controls/agent", json={"value": "off"}, headers=headers).status_code == 200
    for name, value in (("writes", "on"), ("agent", "on")):
        got = http.put(f"/v1/controls/{name}", json={"value": value}, headers=headers)
        assert got.status_code == 403, (name, got.text)
    assert (
        http.put("/v1/controls/writes", json={"value": "nonsense"}, headers=headers).status_code
        == 403
    )


def test_an_admin_and_the_service_key_can_start_things_again(http: TestClient) -> None:
    headers = sign_in(http, "maya", "admin")
    http.put("/v1/controls/writes", json={"value": "frozen"}, headers=headers)
    assert http.put("/v1/controls/writes", json={"value": "on"}, headers=headers).status_code == 200
    http.put("/v1/controls/agent", json={"value": "off"}, headers=headers)
    assert http.put("/v1/controls/agent", json={"value": "on"}, headers=headers).status_code == 200
    key_call = {"value": "on", "reviewer": "ops"}
    assert http.put("/v1/controls/writes", json=key_call, headers=AUTH).status_code == 200


def test_the_control_change_is_recorded_under_the_signed_in_name(http: TestClient) -> None:
    headers = sign_in(http, "omar")
    http.put("/v1/controls/agent", json={"value": "off", "reason": "odd answers"}, headers=headers)
    [agent] = [c for c in http.get("/v1/controls").json()["controls"] if c["name"] == "agent"]
    assert agent["updated_by"] == "omar" and agent["reason"] == "odd answers"


# ------------------------------------------------------------------------------ accounts


def test_only_a_signed_in_admin_can_manage_accounts(http: TestClient) -> None:
    accounts(http).create("maya", PW, "admin")
    body = {"username": "zoe", "password": PW, "role": "reviewer"}
    assert http.post("/v1/admin/reviewers", json=body).status_code == 401
    assert http.post("/v1/admin/reviewers", json=body, headers=AUTH).status_code == 403
    assert http.get("/v1/admin/reviewers", headers=AUTH).status_code == 403

    reviewer = sign_in(http, "omar")
    assert http.get("/v1/admin/reviewers").status_code == 403
    assert http.post("/v1/admin/reviewers", json=body, headers=reviewer).status_code == 403
    http.post("/v1/admin/logout")

    admin = sign_in(http, "maya")
    assert http.post("/v1/admin/reviewers", json=body).status_code == 403  # no csrf header
    made = http.post("/v1/admin/reviewers", json=body, headers=admin)
    assert made.status_code == 201 and made.json()["username"] == "zoe"
    assert "password" not in made.text and "hash" not in made.text
    names = [r["username"] for r in http.get("/v1/admin/reviewers").json()["reviewers"]]
    assert names == ["maya", "omar", "zoe"]


def test_account_errors_are_reported_plainly(http: TestClient) -> None:
    admin = sign_in(http, "maya", "admin")
    weak = {"username": "zoe", "password": "short"}
    assert http.post("/v1/admin/reviewers", json=weak, headers=admin).status_code == 422
    dup = {"username": "maya", "password": PW}
    assert http.post("/v1/admin/reviewers", json=dup, headers=admin).status_code == 422
    badrole = {"username": "zoe", "password": PW, "role": "root"}
    assert http.post("/v1/admin/reviewers", json=badrole, headers=admin).status_code == 422
    assert (
        http.put(
            "/v1/admin/reviewers/ghost/active", json={"active": False}, headers=admin
        ).status_code
        == 404
    )
    assert (
        http.put(
            "/v1/admin/reviewers/ghost/password", json={"password": PW}, headers=admin
        ).status_code
        == 404
    )


def test_an_admin_can_reset_a_password_and_disable_an_account(http: TestClient) -> None:
    admin = sign_in(http, "maya", "admin")
    accounts(http).create("omar", PW)
    other = TestClient(http.app)
    other.post("/v1/admin/login", json={"username": "omar", "password": PW})
    assert other.get("/v1/approvals").status_code == 200

    new = "a different passphrase"
    got = http.put("/v1/admin/reviewers/omar/password", json={"password": new}, headers=admin)
    assert got.status_code == 204
    assert other.get("/v1/approvals").status_code == 401  # signed out everywhere
    assert (
        other.post("/v1/admin/login", json={"username": "omar", "password": new}).status_code == 200
    )

    off = http.put("/v1/admin/reviewers/omar/active", json={"active": False}, headers=admin)
    assert off.status_code == 204 and other.get("/v1/approvals").status_code == 401
    listed = {
        r["username"]: r["active"] for r in http.get("/v1/admin/reviewers").json()["reviewers"]
    }
    assert listed == {"maya": True, "omar": False}


def test_the_last_admin_cannot_lock_everyone_out(http: TestClient) -> None:
    admin = sign_in(http, "maya", "admin")
    got = http.put("/v1/admin/reviewers/maya/active", json={"active": False}, headers=admin)
    assert got.status_code == 422 and "last active admin" in got.text
    assert http.get("/v1/approvals").status_code == 200


# ------------------------------------------------------------------------------ ticket list


def add_tickets(factory: sessionmaker[Session], count: int, at: datetime, customer: str) -> None:
    with factory() as session:
        for _ in range(count):
            session.add(
                Ticket(
                    id=uuid.uuid4(),
                    customer_id=customer,
                    status=TicketStatus.RECEIVED,
                    created_at=at,
                )
            )
        session.commit()


def test_the_ticket_list_pages_through_tickets_created_in_the_same_second(
    seeded_factory: sessionmaker[Session],
) -> None:
    web = build(seeded_factory)
    add_tickets(seeded_factory, 5, T0, "C-0001")
    seen: list[str] = []
    cursor = ""
    for _ in range(10):
        page = web.get(f"/v1/tickets?limit=2{cursor}", headers=AUTH).json()
        seen += [t["id"] for t in page["tickets"]]
        if page["next_before"] is None:
            break
        cursor = f"&before={page['next_before']}"
    assert len(seen) == 5 and len(set(seen)) == 5  # none skipped, none repeated


def test_the_ticket_list_filters_and_shows_the_last_outcome(
    seeded_factory: sessionmaker[Session],
) -> None:
    web = build(seeded_factory)
    add_tickets(seeded_factory, 2, T0 - timedelta(days=1), "C-0009")
    made = open_request(web)
    everything = web.get("/v1/tickets", headers=AUTH).json()["tickets"]
    assert len(everything) == 3 and everything[0]["id"] == str(made)
    assert everything[0]["last_run_outcome"] == "awaiting_approval"
    assert {t["last_run_outcome"] for t in everything[1:]} == {None}
    mine = web.get("/v1/tickets?customer_id=C-0004", headers=AUTH).json()["tickets"]
    assert [t["id"] for t in mine] == [str(made)]
    waiting = web.get("/v1/tickets?status=received", headers=AUTH).json()["tickets"]
    assert len(waiting) == 2 and all(t["status"] == "received" for t in waiting)


def test_the_ticket_list_refuses_bad_input(http: TestClient) -> None:
    for query in (
        "status=bogus",
        "before=garbage",
        "before=2026-01-01T00:00:00_notauuid",
        "limit=0",
        "limit=101",
    ):
        assert http.get(f"/v1/tickets?{query}", headers=AUTH).status_code == 422, query
    assert http.get("/v1/tickets").status_code == 401


def test_the_interactive_docs_can_be_switched_off(seeded_factory: sessionmaker[Session]) -> None:
    on = build(seeded_factory)
    for path in ("/docs", "/redoc", "/openapi.json"):
        assert on.get(path).status_code == 200, path
    off = build(seeded_factory)
    state = off.app.state.kartly  # type: ignore[attr-defined]
    state.docs_enabled = False
    from kartly.api import create_app

    closed = TestClient(create_app(state))
    for path in ("/docs", "/redoc", "/openapi.json"):
        assert closed.get(path).status_code == 404, path
    assert closed.get("/admin").status_code == 200  # the admin page is not API documentation
    assert closed.get("/healthz").status_code == 200
