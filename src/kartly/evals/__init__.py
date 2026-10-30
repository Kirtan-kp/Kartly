"""Golden-set evaluation: cases, a state-based grader and a runner (DESIGN.md section 9)."""

from kartly.evals.cases import Case, CaseFileError, load_cases
from kartly.evals.grader import Check, Evidence, ToolCallEvidence, grade
from kartly.evals.runner import CaseResult, Summary, run_case, run_suite, summarize

__all__ = [
    "Case",
    "CaseFileError",
    "CaseResult",
    "Check",
    "Evidence",
    "Summary",
    "ToolCallEvidence",
    "grade",
    "load_cases",
    "run_case",
    "run_suite",
    "summarize",
]
