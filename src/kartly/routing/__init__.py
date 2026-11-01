"""Routing: which tier a ticket belongs to, decided before any model runs (DESIGN.md section 6)."""

from kartly.routing.rules import RuleVerdict, route_by_rules

__all__ = ["RuleVerdict", "route_by_rules"]
