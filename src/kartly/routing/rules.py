"""Layer 1 of routing: hard rules that force Tier 3 (never autonomous).

Legal threats, fraud and chargebacks, account security, safety complaints and a customer in
distress must reach a human, and the rules must not depend on a model, so they are keyword
patterns. They are tuned for recall: a false alarm costs a human a minute, a miss costs a customer
a bot's answer to a legal threat. The eval in ``kartly.evals.routing`` measures both, and lists the
phrasings the rules are known to miss; the classifier layer (later) exists to catch those.

The rules can only raise a ticket's tier. A ticket no rule fires on is not "safe", it is simply
not decided here (``tier`` is 0 meaning "no rule fired").
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from kartly.guards.screen import normalise_text

TIER3 = 3

_RULES: dict[str, re.Pattern[str]] = {
    "legal": re.compile(
        r"\b(lawyers?|attorneys?|advocates?|solicitors?|sue|suing|sued|lawsuits?|"
        r"legal (action|notice|proceedings?|case|team|department|recourse|remedy)|"
        r"take legal|court|consumer (court|forum|commission|protection)|ncdrc|"
        r"police|cyber ?crime|file a (case|complaint) against|defamation|"
        r"breach of contract)\b"
    ),
    "fraud": re.compile(
        r"\b(chargebacks?|charge ?backs?|fraud(ulent|ster|sters)?|scams?|scammed|scammers?|"
        r"unauthori[sz]ed (transaction|charge|payment|purchase|order)s?|"
        r"stolen (card|credit card|debit card|upi)|"
        r"(dispute|disputing|disputed)\b[^.\n]{0,25}\b(charge|payment|transaction|bank|card)|"
        r"(did not|didn'?t|never) (make|place|authori[sz]ed?)[^.\n]{0,25}"
        r"\b(purchase|payment|order|transaction))\b"
    ),
    "account_security": re.compile(
        r"\b(hacked|hackers?|hacking|compromised|account takeover|phishing|data breach|"
        r"(someone|somebody|anyone) (else )?(logged|got|broke) into|"
        r"(someone|somebody) (else )?(has |had )?"
        r"(accessed|used) my (account|card)|"
        r"(delete|erase|remove|close)\b[^.\n]{0,20}\b(account|personal (data|information))|"
        r"leak(ed)? my (data|details|information)|gdpr|data protection|right to be forgotten)\b"
    ),
    "safety": re.compile(
        r"\b(injur(y|ed|ies)|hurt (me|my|him|her|myself|someone|a child|the (baby|child|kids?))|"
        r"burn(t|ed|s|ing)? (my|his|her|the|a|me)\b|caught fire|on fire|fire hazard|"
        r"exploded|explosion|electric(al)? shock|electrocut\w*|shocked me|"
        r"allergic reaction|rash(?! of\b)|poison\w*|chok(e|ed|ing)|hospital|ambulance|bleeding|"
        r"unsafe|dangerous|swallowed)\b"
    ),
    "distress": re.compile(
        r"\b(suicid\w*|kill (my|him|her)self|end (my life|it all)|self[- ]harm|"
        r"want to die|do(n'?t| not) want to live|no reason to live)\b"
    ),
    "threat": re.compile(
        r"\b(kill|bomb|shoot|stab|burn down|blow up)\b[^.\n]{0,15}"
        r"\b(you|your (staff|team|office|employees|warehouse|company|"
        r"(delivery|courier) (guy|boy|man|person|partner|agent|executive)))\b"
    ),
}


@dataclass(frozen=True)
class RuleVerdict:
    tier: int  # 3 when any rule fired, otherwise 0 ("no rule fired")
    categories: tuple[str, ...] = ()


def route_by_rules(text: str) -> RuleVerdict:
    """The categories whose rules match ``text``, in a stable order; tier 3 if there are any."""
    normalised = normalise_text(text)
    hits = tuple(name for name, pattern in _RULES.items() if pattern.search(normalised))
    return RuleVerdict(TIER3 if hits else 0, hits)
