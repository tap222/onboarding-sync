"""Authentication, authorization, document ACL and metadata-filtered search."""
import json

import pytest

from conftest import AUDITOR, IT, MANAGER, NOBODY, PEOPLE_OPS, PEOPLE_OPS_2, RECRUITER, hire, ingest
from hiresync import tools as toolreg
from hiresync.auth import AuthError, Forbidden, authenticate, issue_dev_token, service_principal
from hiresync.graphql_schema import schema

OFFER = "Offer Letter\nLegal First Name: Priya\nBase salary: $185,000\nStart Date: December 14, 2026\n"
IT_FORM = "IT Hardware Form\nStreet: 1 Oak\nCity: Austin\nZIP: 78701\nLaptop: MacBook Pro 16\n"


async def call(app, principal, tool, **args):
    out, is_error = await toolreg.execute(app, tool, args, principal=principal)
    return json.loads(out), is_error


@pytest.fixture
async def seeded(app):
    await ingest(app, "H1", hire(), "e1")                                     # managed by dana
    await ingest(app, "H2", hire("Jonas", "Weber", manager_email="sam@betterup.co"), "e2")
    offer = await app.documents.upload("H1", "offer.txt", OFFER.encode(), "text/plain", RECRUITER, "offer_letter")
    form = await app.documents.upload("H1", "it.txt", IT_FORM.encode(), "text/plain", PEOPLE_OPS, "it_hardware_form")
    await app.drain()
    return app, offer["doc_hash"], form["doc_hash"]


# ------------------------------------------------------------ authentication
def test_tokens_verify_and_users_cannot_claim_service_groups():
    p = authenticate("Bearer " + issue_dev_token("u1", "a@betterup.co", ["recruiting", "svc-triage"]))
    assert p.groups == {"recruiting"}
    with pytest.raises(AuthError):
        authenticate(None)
    with pytest.raises(AuthError):
        authenticate("Bearer " + issue_dev_token("u1", "a@b.co", ["people-ops"], ttl_s=-10))  # expired
    forged = issue_dev_token("u1", "a@b.co", ["people-ops"])[:-4] + "AAAA"
    with pytest.raises(AuthError):
        authenticate("Bearer " + forged)


# ------------------------------------------------------- hire-level access
async def test_hiring_manager_sees_only_own_hires_and_no_pii(seeded):
    app, *_ = seeded
    own, err = await call(app, MANAGER, "get_hire", hire_id="H1")
    assert not err and own["record"]["shipping_address"] == "[MASKED]" and own["record"]["personal_email"] == "[MASKED]"
    other, err = await call(app, MANAGER, "get_hire", hire_id="H2")
    assert err and "forbidden" in other["error"]


async def test_unknown_and_forbidden_hires_look_the_same(seeded):
    app, *_ = seeded
    a, _ = await call(app, MANAGER, "get_hire", hire_id="H2")
    b, _ = await call(app, MANAGER, "get_hire", hire_id="H-404")
    assert a == b  # no existence oracle


async def test_agent_is_scoped_to_its_hire(seeded):
    app, *_ = seeded
    svc = service_principal("svc-triage", "H1")
    _, err = await call(app, svc, "get_hire", hire_id="H1")
    assert not err
    out, err = await call(app, svc, "get_hire", hire_id="H2")
    assert err and "scoped to H1" in out["error"]


# ---------------------------------------------------------- document ACL
@pytest.mark.parametrize("who,offer_ok,form_ok", [
    (PEOPLE_OPS, True, True),
    (RECRUITER, True, True),     # uploaded the offer; internal docs readable
    (IT, False, True),           # no compensation documents for IT
    (MANAGER, False, False),     # no docs for managers
    (AUDITOR, False, False),
])
async def test_document_access_matrix(seeded, who, offer_ok, form_ok):
    app, offer, form = seeded
    _, err1 = await call(app, who, "get_document", doc_hash=offer)
    _, err2 = await call(app, who, "get_document", doc_hash=form)
    assert (not err1, not err2) == (offer_ok, form_ok)


async def test_search_filters_by_acl_and_metadata(seeded):
    app, offer, form = seeded
    it_hits, _ = await call(app, IT, "search_documents", query="base salary compensation", k=10)
    assert it_hits and all(h["classification"] != "confidential" for h in it_hits)
    po_hits, _ = await call(app, PEOPLE_OPS, "search_documents", query="base salary compensation", k=10)
    assert po_hits[0]["doc_hash"] == offer and "185,000" in po_hits[0]["text"]
    typed, _ = await call(app, PEOPLE_OPS, "search_documents", query="salary", doc_types=["it_hardware_form"])
    assert {h["doc_hash"] for h in typed} == {form}
    h = po_hits[0]
    full = app.store.get_document(offer)["text_redacted"]
    assert full[h["citation"]["char_start"]:h["citation"]["char_end"]] in h["text"]


async def test_provenance_hides_citations_to_unreadable_documents(seeded):
    app, offer, _ = seeded
    rows, _ = await call(app, IT, "get_field_provenance", hire_id="H1", path="start_date")
    cited = [r for r in rows if r["doc_hash"] == offer]
    assert cited and cited[0]["citation"] == {"restricted": True, "classification": "confidential"}


async def test_duplicate_upload_reveals_nothing_without_read_access(seeded):
    app, offer, _ = seeded
    it_user = IT.__class__("u-it2", "it2@betterup.co", frozenset({"it", "recruiting"}) - {"recruiting"})
    with pytest.raises(Forbidden):  # IT can't upload at all
        await app.documents.upload("H1", "x.txt", OFFER.encode(), "text/plain", it_user, "offer_letter")
    r = await app.documents.upload("H1", "copy.txt", OFFER.encode(), "text/plain", PEOPLE_OPS_2, "offer_letter")
    assert r["duplicate"] and r["doc_hash"] == offer  # people-ops may read it: full view


async def test_document_cannot_be_filed_under_a_second_hire(seeded):
    app, *_ = seeded
    from hiresync.guardrails import GuardrailViolation
    with pytest.raises(GuardrailViolation, match="different hire"):
        await app.documents.upload("H2", "offer.txt", OFFER.encode(), "text/plain", RECRUITER, "offer_letter")


# ----------------------------------------------------------- approvals
async def test_separation_of_duties_on_approvals(app):
    await ingest(app, "H1", hire(), "e1")
    app.systems["okta"].activate("H1")
    await app.submit_change("H1", {"legal_first_name": "Pri"}, source="human", actor=PEOPLE_OPS.email, event_id=None)
    await app.drain()
    apr = app.store.list_approvals(status="pending")[0]
    with pytest.raises(Forbidden, match="another approver"):
        await app.orchestrator.decide(apr["approval_id"], "approved", PEOPLE_OPS)
    with pytest.raises(Forbidden):
        await app.orchestrator.decide(apr["approval_id"], "approved", RECRUITER)  # no approval:decide
    out = await app.orchestrator.decide(apr["approval_id"], "approved", PEOPLE_OPS_2)
    assert out["status"] == "approved"


# ------------------------------------------------------- per-caller tools
def test_tool_lists_differ_by_group():
    names = lambda p: {t.name for t in toolreg.tools_for(p)}
    assert "decide_approval" in names(PEOPLE_OPS) and "decide_approval" not in names(RECRUITER)
    assert "search_documents" not in names(MANAGER) and names(NOBODY) == set()


async def test_graphql_enforces_the_same_rules(seeded):
    app, offer, _ = seeded
    q = '{ hires { hireId } hire(hireId:"H1"){ current documents { docHash classification } } }'
    res = await schema.execute(q, context_value={"app": app, "principal": MANAGER})
    assert res.errors is None and [h["hireId"] for h in res.data["hires"]] == ["H1"]
    assert res.data["hire"]["current"]["shipping_address"] == "[MASKED]" and res.data["hire"]["documents"] == []
    res = await schema.execute('{ hire(hireId:"H2"){ currentVersion } }', context_value={"app": app, "principal": MANAGER})
    assert res.errors and "forbidden" in res.errors[0].message
    res = await schema.execute(f'{{ document(docHash:"{offer}"){{ docHash }} }}', context_value={"app": app, "principal": IT})
    assert res.data["document"] is None
