"""The admin page driven in a real browser (Chromium via Playwright).

These tests catch what string-level checks cannot: a policy that blocks the page's own styles, a
list printed as "[object HTMLDivElement]", HTML in a customer message being run instead of shown.
They are skipped when Playwright or its browser is not installed (``playwright install chromium``).
"""

from __future__ import annotations

import socket
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
import uvicorn
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from kartly.store.models import Base, Message, MessageRole, Ticket
from tests.support import seeded_session_factory
from tests.unit.test_api_admin import PW, build
from tests.unit.test_api_approvals import AUTH, open_request

sync_api = pytest.importorskip("playwright.sync_api")

pytestmark = pytest.mark.browser
HOSTILE = '<img src=x onerror="window.__pwned=1"><script>window.__pwned=2</script> please refund'
# Failed requests are logged by the browser as console errors; these are expected here (the page
# asks before signing in, and a server without a queue or alerts answers 404 or 503 on purpose).
EXPECTED_NOISE = ("Failed to load resource",)


class Site:
    def __init__(self, url: str, factory: sessionmaker[Session], ticket_id: str) -> None:
        self.url, self.factory, self.ticket_id = url, factory, ticket_id


@pytest.fixture(scope="module")
def site(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Site]:
    db = Path(tmp_path_factory.mktemp("browser")) / "site.db"
    engine = create_engine(f"sqlite:///{db}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    factory = seeded_session_factory(engine)
    web = build(factory)
    ticket_id = str(open_request(web))
    with factory() as session:
        ticket = session.query(Ticket).one()
        session.add(Message(ticket_id=ticket.id, role=MessageRole.CUSTOMER, content=HOSTILE))
        session.commit()
    accounts = web.app.state.kartly.reviewers  # type: ignore[attr-defined]
    accounts.create("maya", PW, "admin")
    accounts.create("omar", PW, "reviewer")

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(web.app, host="127.0.0.1", port=port, log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{port}"
    for _ in range(100):
        try:
            if httpx.get(url + "/healthz").status_code == 200:
                break
        except httpx.TransportError:
            time.sleep(0.1)
    else:
        pytest.fail("the test server did not start")
    yield Site(url, factory, ticket_id)
    server.should_exit = True
    thread.join(timeout=10)
    engine.dispose()


@pytest.fixture(scope="module")
def browser() -> Iterator[Any]:
    with sync_api.sync_playwright() as pw:
        try:
            chromium = pw.chromium.launch()
        except Exception as exc:  # no browser binary installed
            pytest.skip(f"Chromium is not available: {exc}")
        yield chromium
        chromium.close()


class Visit:
    """One browser page that records anything the browser itself complains about."""

    def __init__(self, browser: Any, site: Site) -> None:
        self.site = site
        self.page = browser.new_page(viewport={"width": 1200, "height": 900})
        self.problems: list[str] = []
        self.page.on("pageerror", lambda e: self.problems.append(f"exception: {e}"))
        self.page.on("console", self._console)

    def _console(self, message: Any) -> None:
        if message.type == "error" and not any(n in message.text for n in EXPECTED_NOISE):
            self.problems.append(f"console: {message.text}")

    def sign_in(self, name: str, password: str = PW) -> None:
        self.page.goto(self.site.url + "/admin")
        self.page.fill("#u", name)
        self.page.fill("#p", password)
        self.page.click("button[type=submit]")

    def open(self, tab: str) -> None:
        self.page.click(f'nav button:has-text("{tab}")')
        self.page.wait_for_timeout(500)

    def text(self) -> str:
        return str(self.page.text_content("main"))


@pytest.fixture
def visit(browser: Any, site: Site) -> Iterator[Visit]:
    v = Visit(browser, site)
    yield v
    v.page.close()


def test_a_wrong_password_is_refused_and_a_right_one_signs_in(visit: Visit) -> None:
    visit.sign_in("maya", "not the password")
    visit.page.wait_for_selector(".notice.bad")
    assert "invalid username or password" in str(visit.page.text_content(".notice.bad"))
    assert visit.page.query_selector("nav") is None
    visit.page.fill("#p", PW)
    visit.page.click("button[type=submit]")
    visit.page.wait_for_selector("nav")
    assert "maya (admin)" in str(visit.page.text_content("nav"))
    assert visit.problems == []


def test_html_in_a_customer_message_is_shown_as_text_and_never_runs(visit: Visit) -> None:
    visit.sign_in("omar")
    visit.page.wait_for_selector("nav")
    visit.open("Tickets")
    visit.page.click("tr.click")
    visit.page.wait_for_selector(".msg.customer")
    visit.page.wait_for_timeout(300)
    assert '<img src=x onerror="window.__pwned=1">' in visit.text()
    assert visit.page.evaluate("window.__pwned") is None
    assert visit.page.query_selector("main img") is None
    assert visit.page.query_selector("main script") is None
    assert visit.problems == []


def test_a_ticket_shows_its_messages_runs_and_trace_as_real_elements(visit: Visit) -> None:
    visit.sign_in("omar")
    visit.page.wait_for_selector("nav")
    visit.open("Tickets")
    visit.page.click("tr.click")
    visit.page.wait_for_selector(".msg")
    assert "[object" not in visit.text()
    assert visit.page.locator(".msg").count() >= 2
    visit.page.click('button:has-text("Trace")')
    visit.page.wait_for_selector(".trace details")
    assert "[object" not in visit.text()
    assert visit.page.locator(".trace details").count() >= 1
    assert visit.problems == []


def test_every_tab_loads_without_the_browser_reporting_a_problem(visit: Visit) -> None:
    visit.sign_in("maya")
    visit.page.wait_for_selector("nav")
    for tab in (
        "Overview",
        "Costs",
        "Tickets",
        "Approvals",
        "Dead letters",
        "Alerts",
        "Controls",
        "Accounts",
    ):
        visit.open(tab)
        assert visit.text().strip(), tab
        assert "[object" not in visit.text(), tab
        assert visit.page.query_selector(".notice.bad") is None or tab in (
            "Dead letters",
            "Alerts",
        ), tab
    assert visit.problems == []


def test_a_reviewer_cannot_see_accounts_or_turn_things_back_on(visit: Visit) -> None:
    visit.sign_in("omar")
    visit.page.wait_for_selector("nav")
    assert visit.page.query_selector('nav button:has-text("Accounts")') is None
    visit.open("Controls")
    assert "An admin turns them back on" in visit.text()
    assert visit.page.locator('button:has-text("Set frozen")').is_enabled()
    assert visit.problems == []


def test_a_reviewer_approves_a_request_from_the_page(visit: Visit, site: Site) -> None:
    visit.sign_in("omar")
    visit.page.wait_for_selector("nav")
    visit.open("Approvals")
    visit.page.wait_for_selector(".card button:has-text('Approve')")
    visit.page.fill("textarea", "checked the order")
    visit.page.click("button:has-text('Approve')")
    visit.page.wait_for_selector(".notice")
    with httpx.Client(base_url=site.url, headers=AUTH) as api:
        mine = api.get("/v1/approvals?status=approved").json()["approvals"]
    assert [a["decided_by"] for a in mine] == ["human:omar"]
    assert visit.problems == []


def test_signing_out_returns_to_the_login_form(visit: Visit) -> None:
    visit.sign_in("omar")
    visit.page.wait_for_selector("nav")
    visit.page.click('button:has-text("Sign out")')
    visit.page.wait_for_selector("form.login")
    visit.page.reload()
    visit.page.wait_for_selector("form.login")  # the cookie is gone, so a reload stays signed out
    assert visit.problems == []
