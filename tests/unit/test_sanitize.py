from __future__ import annotations

import json
from typing import Any

from hypothesis import given, settings
from hypothesis import strategies as st

from kartly.sanitize import MAX_DEPTH, MAX_KEYS, MAX_LIST, MAX_STR, clean_str, safe_json


def test_long_strings_are_truncated() -> None:
    cleaned = clean_str("x" * 10_000)
    assert len(cleaned) == MAX_STR + 1
    assert cleaned.endswith("…")
    assert clean_str("short") == "short"


def test_nul_and_lone_surrogates_are_removed() -> None:
    cleaned = clean_str("a\x00b\ud800c")
    assert "\x00" not in cleaned
    cleaned.encode("utf-8")  # must be encodable: Postgres rejects anything else


def test_non_finite_floats_and_huge_ints_become_strings() -> None:
    assert safe_json(float("nan")) == "nan"
    assert safe_json(float("inf")) == "inf"
    assert safe_json(10**5000) == "<big int>"
    assert safe_json(12345) == 12345
    assert safe_json(1.5) == 1.5


def test_containers_are_bounded() -> None:
    wide = {f"k{i}": i for i in range(100)}
    assert len(safe_json(wide)) == MAX_KEYS
    assert len(safe_json(list(range(100)))) == MAX_LIST
    deep: Any = "leaf"
    for _ in range(10):
        deep = {"x": deep}
    depth = 0
    node = safe_json(deep)
    while isinstance(node, dict):
        node = node["x"]
        depth += 1
    assert depth == MAX_DEPTH
    assert isinstance(node, str) and node.startswith("<")


def test_unknown_objects_are_described_not_serialised() -> None:
    assert safe_json(object()) == "<object>"
    assert safe_json({1, 2}) == "<set>"


scalars = (
    st.none()
    | st.booleans()
    | st.integers(min_value=-(10**400), max_value=10**400)
    | st.floats()
    | st.text()
)
anything = st.recursive(
    scalars,
    lambda children: st.lists(children, max_size=4) | st.dictionaries(st.text(), children),
    max_leaves=25,
)


@settings(max_examples=200, deadline=None)
@given(anything)
def test_any_value_becomes_strict_json(value: Any) -> None:
    dumped = json.dumps(safe_json(value), allow_nan=False)  # no exception, no NaN
    assert "\\u0000" not in dumped


def test_trace_limits_keep_far_more_than_audit_limits() -> None:
    from kartly.sanitize import TRACE_LIMITS

    long_text = "x" * 3000
    assert len(safe_json(long_text)) == 201  # audit default: 200 characters plus an ellipsis
    assert safe_json(long_text, TRACE_LIMITS) == long_text
    assert len(safe_json(list(range(100)))) == 10
    assert len(safe_json(list(range(100)), TRACE_LIMITS)) == 50
    deep = {"a": {"b": {"c": {"d": {"e": 1}}}}}
    assert safe_json(deep)["a"]["b"]["c"] == "<dict>"
    assert safe_json(deep, TRACE_LIMITS)["a"]["b"]["c"]["d"]["e"] == 1


def test_trace_limits_still_strip_what_postgres_rejects() -> None:
    from kartly.sanitize import TRACE_LIMITS

    cleaned = safe_json({"s": "a\x00b\ud800", "n": float("nan"), "big": 10**5000}, TRACE_LIMITS)
    assert cleaned == {"s": "ab?", "n": "nan", "big": "<big int>"}
