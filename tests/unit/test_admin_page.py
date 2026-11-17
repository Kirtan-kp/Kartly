"""The admin page is served with a policy that forbids anything it does not ship itself."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session, sessionmaker

from kartly.api.app import STATIC_DIR
from tests.unit.test_api_admin import build


@pytest.fixture
def http(seeded_factory: sessionmaker[Session]) -> TestClient:
    return build(seeded_factory)


@pytest.mark.parametrize(
    ("path", "kind"),
    [
        ("/admin", "text/html"),
        ("/admin/admin.js", "text/javascript"),
        ("/admin/admin.css", "text/css"),
    ],
)
def test_the_page_and_its_files_are_served_without_signing_in(
    http: TestClient, path: str, kind: str
) -> None:
    got = http.get(
        path
    )  # the page asks for credentials itself; the data routes are what is guarded
    assert got.status_code == 200 and got.headers["content-type"].startswith(kind)
    assert got.headers["x-content-type-options"] == "nosniff"
    assert got.headers["referrer-policy"] == "no-referrer"


def test_the_policy_allows_only_our_own_files(http: TestClient) -> None:
    policy = http.get("/admin").headers["content-security-policy"]
    directives = {d.split()[0]: d.split()[1:] for d in policy.split("; ")}
    assert directives["default-src"] == ["'none'"]
    assert directives["script-src"] == ["'self'"]
    assert directives["frame-ancestors"] == ["'none'"]
    assert "'unsafe-inline'" not in policy and "'unsafe-eval'" not in policy


def test_the_html_has_no_inline_script_or_style_for_the_policy_to_block() -> None:
    html = (STATIC_DIR / "admin.html").read_text()
    assert not re.search(r"<script(?![^>]*\bsrc=)", html)
    assert "<style" not in html and "onclick=" not in html
    assert not re.search(r"(src|href)=\"https?://", html)


def test_the_script_sets_no_inline_styles_because_the_policy_blocks_them() -> None:
    script = (STATIC_DIR / "admin.js").read_text()
    assert "style:" not in script and 'setAttribute("style"' not in script
    assert "style=" not in (STATIC_DIR / "admin.html").read_text()


def test_several_children_are_never_passed_straight_to_replace_children() -> None:
    """It does not flatten arrays; the page once printed "[object HTMLDivElement]" because of it."""
    script = (STATIC_DIR / "admin.js").read_text()
    for call in re.findall(r"replaceChildren\(([^;]*?)\);", script, flags=re.S):
        assert "map(" not in call and ",\n" not in call, call[:60]


def test_the_script_never_writes_server_text_as_html() -> None:
    script = (STATIC_DIR / "admin.js").read_text()
    for banned in (
        "innerHTML",
        "outerHTML",
        "insertAdjacentHTML",
        "document.write",
        "eval(",
        "new Function",
    ):
        assert banned not in script, banned


def test_the_script_is_valid_javascript() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    done = subprocess.run(
        [node, "--check", str(Path(STATIC_DIR / "admin.js"))], capture_output=True, text=True
    )
    assert done.returncode == 0, done.stderr


def test_every_route_the_page_calls_exists(http: TestClient) -> None:
    script = (STATIC_DIR / "admin.js").read_text()
    wanted = set(re.findall(r"\"(/v1/[a-z/-]+)", script))
    known = {r.path for r in http.app.routes}  # type: ignore[attr-defined]
    for path in wanted:
        assert any(
            k == path or k.startswith(path.rstrip("/") + "/") or k.startswith(path) for k in known
        ), path


def test_the_page_is_not_in_the_api_documentation(http: TestClient) -> None:
    assert "/admin" not in http.get("/openapi.json").json()["paths"]
