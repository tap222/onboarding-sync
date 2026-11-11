import json

from fastapi.testclient import TestClient

import hiresync.api as api_mod
from conftest import make_app
from hiresync.auth import issue_dev_token

REC = {"legal_first_name": "Ana", "legal_last_name": "Silva", "personal_email": "a@example.com",
       "start_date": "2027-03-01", "job_title": "Designer", "department": "Design",
       "manager_email": "m@betterup.co", "work_location": {"country": "US"},
       "shipping_address": {"line1": "5 Main", "city": "Reno", "state": "NV", "postal_code": "89501", "country": "US"}}


def signed(body: dict) -> tuple[bytes, dict]:
    raw = json.dumps(body).encode()
    return raw, {"Ashby-Signature": api_mod.ashby_signature(raw), "Content-Type": "application/json"}


def test_webhook_signature_idempotency_and_graphql_auth(tmp_path):
    api_mod.app_state = make_app(str(tmp_path / "h.db"))
    with TestClient(api_mod.api) as c:
        raw, headers = signed({"hire_id": "H-7", "patch": REC})
        for _ in range(2):  # same Idempotency-Key twice
            r = c.post("/webhooks/ashby", content=raw, headers={**headers, "Idempotency-Key": "k1"})
            assert r.status_code == 200
        assert r.json()["duplicate"] is True
        bad = c.post("/webhooks/ashby", content=raw, headers={"Ashby-Signature": "nope", "Idempotency-Key": "k9"})
        assert bad.status_code == 401
        stale = api_mod.ashby_signature(raw, ts=1_000_000_000)  # valid HMAC, replayed years later
        r = c.post("/webhooks/ashby", content=raw, headers={"Ashby-Signature": stale, "Idempotency-Key": "k8",
                                                            "Content-Type": "application/json"})
        assert r.status_code == 401
        raw2, h2 = signed({"hire_id": "H-8", "patch": {"legal_first_name": "X"}})
        assert c.post("/webhooks/ashby", content=raw2, headers={**h2, "Idempotency-Key": "k2"}).status_code == 422

        q = {"query": '{ me { groups } hire(hireId:"H-7"){ currentVersion } tools { name } }'}
        assert c.post("/graphql", json=q).status_code == 401  # no token
        tok = issue_dev_token("u-maya", "maya@betterup.co", ["people-ops"])
        data = c.post("/graphql", json=q, headers={"Authorization": f"Bearer {tok}"}).json()["data"]
        assert data["me"]["groups"] == ["people-ops"] and data["hire"]["currentVersion"] == 1
        assert {"name": "decide_approval"} in data["tools"]
    api_mod.app_state = None
