from datetime import date

import pytest

from hiresync.app import HireSync
from hiresync.auth import Principal

TODAY = date(2026, 11, 10)

PEOPLE_OPS = Principal("u-maya", "maya@betterup.co", frozenset({"people-ops"}))
PEOPLE_OPS_2 = Principal("u-lee", "lee@betterup.co", frozenset({"people-ops"}))
RECRUITER = Principal("u-raj", "raj@betterup.co", frozenset({"recruiting"}))
IT = Principal("u-ivy", "ivy@betterup.co", frozenset({"it"}))
MANAGER = Principal("u-dana", "dana@betterup.co", frozenset({"hiring-manager"}))
AUDITOR = Principal("u-ann", "ann@betterup.co", frozenset({"auditor"}))
NOBODY = Principal("u-x", "x@contractor.io", frozenset())


def hire(first="Priya", last="Raman", **over):
    rec = {
        "legal_first_name": first, "legal_last_name": last, "personal_email": "p@example.com",
        "start_date": "2026-12-07", "job_title": "Engineer", "department": "Engineering",
        "manager_email": "dana@betterup.co", "employment_type": "FTE",
        "work_location": {"country": "US", "city": "Austin", "remote": True},
        "shipping_address": {"line1": "1 Oak", "city": "Austin", "state": "TX", "postal_code": "78701", "country": "US"},
    }
    rec.update(over)
    return rec


def make_app(path: str, **kw) -> HireSync:
    kw.setdefault("today", lambda: TODAY)
    kw.setdefault("agent_mode", "heuristic")
    return HireSync(db_path=path, **kw)


@pytest.fixture
async def app(tmp_path):
    a = make_app(str(tmp_path / "t.db"))
    await a.start()
    yield a
    await a.stop()


async def ingest(app, hid, patch, eid):
    r = await app.submit_change(hid, patch, source="ashby", actor="svc-ashby", event_id=eid)
    await app.drain()
    return r
