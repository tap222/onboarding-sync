"""Cross-system validation. Deterministic rules, owned by code, run before anything
is written anywhere. Claude never decides whether a rule fired - only what to do next.
"""
from __future__ import annotations

import re
from datetime import date
from typing import Optional

from pydantic import ValidationError

from .connectors import SHIP_LEAD_BUSINESS_DAYS, MockSystem, business_days_before, okta_login
from .models import HireRecord, Issue, Severity
from .versioning import FieldChange

POSTAL_PATTERNS = {
    "US": r"^\d{5}(-\d{4})?$",
    "GB": r"^[A-Z]{1,2}\d[A-Z\d]? ?\d[A-Z]{2}$",
    "CA": r"^[A-Z]\d[A-Z] ?\d[A-Z]\d$",
    "IN": r"^\d{6}$",
}
NAME_RE = re.compile(r"^[^\W\d_]+(?:[ '\-.][^\W\d_]+)*$", re.UNICODE)


def validate(
    new: dict,
    old: Optional[dict],
    changes: list[FieldChange],
    systems: dict[str, MockSystem],
    today: date,
) -> list[Issue]:
    issues: list[Issue] = []

    # 1. Schema: the record must be a valid canonical hire at all.
    try:
        HireRecord.model_validate(new)
    except ValidationError as exc:
        for err in exc.errors():
            issues.append(Issue(code="SCHEMA_INVALID", severity=Severity.BLOCK,
                                field=".".join(map(str, err["loc"])), message=err["msg"]))
        return issues  # nothing else is meaningful on a malformed record

    changed = {c.path for c in changes}
    start = date.fromisoformat(str(new["start_date"]))

    # 2. Names: catch typos before they become an email address and a shipping label.
    for f in ("legal_first_name", "legal_last_name", "preferred_first_name"):
        v = new.get(f)
        if v and (not NAME_RE.match(v) or v != v.strip() or "  " in v):
            issues.append(Issue(code="NAME_SUSPICIOUS", severity=Severity.WARN, field=f,
                                message=f"{f}={v!r} contains digits, symbols or stray whitespace",
                                systems=["workday", "okta", "expoit", "tracker"]))
        elif v and (v.isupper() or v.islower()) and len(v) > 2:
            issues.append(Issue(code="NAME_CASING", severity=Severity.WARN, field=f,
                                message=f"{f}={v!r} is all upper/lower case",
                                systems=["workday", "okta", "tracker"]))

    # 3. Start date gates: equipment needs N business days to ship.
    if start < today:
        issues.append(Issue(code="START_DATE_IN_PAST", severity=Severity.BLOCK, field="start_date",
                            message=f"start date {start} is before today {today}"))
    elif business_days_before(start, SHIP_LEAD_BUSINESS_DAYS) < today:
        sev = Severity.BLOCK if "start_date" in changed and old else Severity.WARN
        issues.append(Issue(code="EQUIPMENT_LEAD_TIME", severity=sev, field="start_date",
                            message=f"start {start} leaves under {SHIP_LEAD_BUSINESS_DAYS} business days to ship a laptop",
                            systems=["expoit"]))

    # 4. Location vs shipping address.
    if new["work_location"]["country"] != new["shipping_address"]["country"]:
        issues.append(Issue(code="COUNTRY_MISMATCH", severity=Severity.WARN, field="shipping_address.country",
                            message=f"works in {new['work_location']['country']} but ships to "
                                    f"{new['shipping_address']['country']} (export/tax implications)",
                            systems=["expoit", "workday"]))
    pattern = POSTAL_PATTERNS.get(new["shipping_address"]["country"])
    if pattern and not re.match(pattern, new["shipping_address"]["postal_code"].upper()):
        issues.append(Issue(code="POSTAL_CODE_INVALID", severity=Severity.BLOCK,
                            field="shipping_address.postal_code",
                            message=f"{new['shipping_address']['postal_code']!r} is not a valid "
                                    f"{new['shipping_address']['country']} postal code",
                            systems=["expoit"]))

    # 5. Identity: the derived Okta login is the most expensive thing to get wrong.
    okta = systems.get("okta")
    login = okta_login(new)
    if okta is not None:
        owner = okta.login_owner(login)  # type: ignore[attr-defined]
        if owner and owner != new["hire_id"]:
            issues.append(Issue(code="OKTA_LOGIN_COLLISION", severity=Severity.BLOCK, field="legal_last_name",
                                message=f"{login} already belongs to {owner}", systems=["okta"]))
        existing = okta.records.get(new["hire_id"])
        if existing and existing["login"] != login:
            sev = Severity.BLOCK if existing.get("status") == "ACTIVE" else Severity.WARN
            issues.append(Issue(code="OKTA_LOGIN_CHANGE", severity=sev, field="legal_first_name",
                                message=f"login would change {existing['login']} -> {login} "
                                        f"(account is {existing.get('status')})",
                                systems=["okta", "lumos", "tracker"]))

    # 6. Drift: Workday was edited directly since our last write.
    wd = systems.get("workday")
    if wd is not None and old and new["hire_id"] in wd.records:
        expected_old = wd.project(old, {})
        actual = wd.records[new["hire_id"]]
        drifted = sorted(k for k in expected_old if k != "worker_id" and actual.get(k) != expected_old[k])
        if drifted:
            issues.append(Issue(code="DOWNSTREAM_DRIFT", severity=Severity.WARN, field=",".join(drifted),
                                message=f"workday fields {drifted} were changed outside the pipeline; "
                                        "this change would overwrite them", systems=["workday"]))

    return issues
