"""Prompt versioning, the behavioural-test harness, chunking and HNSW recall."""
import json
import random
from types import SimpleNamespace

import pytest

from conftest import hire, ingest
from hiresync.behave import compare, load_cases
from hiresync.chunking import chunk_text
from hiresync.prompt_registry import PromptRegistry, registry
from hiresync.vectors import HashingEmbedder, HNSWIndex


# --------------------------------------------------------- prompt versions
def test_every_prompt_version_is_locked_and_unchanged():
    status = registry().lock_status()
    assert status == {"changed": [], "unlocked": []}, (
        "a prompt file changed or is new: never edit a version in place - add vN+1 and run "
        "`python -m hiresync.prompt_registry lock`")


def test_editing_a_locked_version_is_detected(tmp_path):
    import shutil
    from hiresync.prompt_registry import PROMPT_DIR
    root = tmp_path / "prompts"
    shutil.copytree(PROMPT_DIR, root)
    f = root / "triage" / "v1.md"
    f.write_text(f.read_text().replace("You triage", "You carefully triage"))
    assert PromptRegistry(root).lock_status()["changed"] == ["triage@v1"]
    with pytest.raises(SystemExit):
        PromptRegistry(root).write_lock()


def test_active_version_override(monkeypatch):
    assert registry().get("triage").version == "v1"
    monkeypatch.setenv("PROMPT_TRIAGE_VERSION", "v2")
    assert PromptRegistry().get("triage").ref == "triage@v2"


def test_sha_covers_model_and_effort():
    pv = registry().get("triage", "v1")
    from dataclasses import replace
    assert replace(pv, effort="high").sha256 != pv.sha256


# ---------------------------------------------------- behavioural harness
class RuleClient:
    """Stands in for Claude so the harness mechanics are testable offline:
    behaves like v1 or v2 depending on the system prompt it is given."""

    def __init__(self):
        self.beta = SimpleNamespace(messages=self)

    async def create(self, **kw):
        system, user = kw["system"], kw["messages"][0]["content"]
        strict_identity = "even with precedent" in system  # only in triage@v2
        situation = json.loads(user.split("\n", 1)[1].split("\n\nSource documents")[0])
        codes = [i["code"] for i in situation["issues"]]
        blocks = [i for i in situation["issues"] if i["severity"] == "block"]
        precedent = "approved" in user.split("Similar past episodes", 1)[1]
        if blocks or (strict_identity and "OKTA_LOGIN_CHANGE" in codes) or not precedent:
            action, conf = "hold_for_approval", 0.9
        else:
            action, conf = "proceed", 0.9
        evidence = [f"issue:{c}" for c in codes]
        if action == "proceed":
            evidence += [f"episode:{e['run_id']}" for e in json.loads(user.split("(most similar first):\n", 1)[1])]
        patch = [{"path": "legal_first_name", "value": "Priya"}] if "NAME_CASING" in codes else []
        body = {"action": action, "confidence": conf, "rationale": f"{codes} considered", "evidence": evidence,
                "proposed_patch": patch, "approver_message": f"{situation['hire_id']}: review"}
        return SimpleNamespace(stop_reason="end_turn", stop_details=None, usage=None,
                               content=[SimpleNamespace(type="text", text=json.dumps(body))])


async def test_behaviour_report_shows_v2_fixes_identity_case():
    rep = await compare(["v1", "v2"], RuleClient(), repeats=2, cases=load_cases())
    v1, v2 = rep["versions"]["v1"], rep["versions"]["v2"]
    assert v1["cases"]["staged_login_change_held_despite_precedent"]["rate"] == 0
    assert v2["cases"]["staged_login_change_held_despite_precedent"]["rate"] == 1
    assert {"case": "staged_login_change_held_despite_precedent", "v1": 0.0, "v2": 1.0, "verdict": "improved"} in rep["changed"]
    assert v1["prompt_sha"] != v2["prompt_sha"] and v2["pass_rate"] == 1.0


# ------------------------------------------------------------- chunking
MIXED = """BETTERUP OFFER LETTER

Legal First Name: Jonas
Legal Last Name: Weber
Start Date: December 14, 2026
Job Title: People Partner
Base Salary: $150,000
Equity: 4,000 RSUs vesting over four years
Signing Bonus: $10,000
Street: 88 Pine Ave
City: Denver
State: CO
ZIP: 80202
Country: United States

## Benefits
- Medical, dental and vision coverage from day one
- 401(k) with a 4% company match
- Flexible time off and 12 company holidays
- $1,500 annual learning stipend

## Travel Policy
Employees book travel through the company portal. Flights over four hours may be booked in premium economy. Hotel stays are reimbursed up to the city rate published by finance. Receipts must be submitted within thirty days.
Expense reports are approved by the manager and audited quarterly by finance. Late submissions delay reimbursement to the next cycle.

The laptop program ships a MacBook to every new hire before the start date. Engineering receives a MacBook Pro 16 while other teams receive a MacBook Air. Devices are enrolled in device management before shipping. Lost devices must be reported to IT within one business day.

| Week | Activity | Owner |
| 1 | Orientation and accounts | People Ops |
| 2 | Team onboarding | Manager |
"""


def _invariants(text, chunks):
    assert all(text[c.start:c.end] == c.text for c in chunks)
    covered = set().union(*(range(c.start, c.end) for c in chunks))
    assert {i for i, ch in enumerate(text) if not ch.isspace()} <= covered


def test_adaptive_chunking_follows_structure():
    from hiresync.vectors import HashingEmbedder
    chunks = chunk_text(MIXED, embed=HashingEmbedder().embed)
    _invariants(MIXED, chunks)
    kinds = [c.kind for c in chunks]
    assert kinds[0] == "kv" and "list" in kinds and "prose" in kinds
    assert {c.heading for c in chunks} >= {"BETTERUP OFFER LETTER", "Benefits", "Travel Policy"}
    for line in MIXED.splitlines():  # a form field is never split from its value
        if ": " in line and not line.startswith(("Employees", "The ")):
            i = MIXED.index(line)
            assert any(c.start <= i and i + len(line) <= c.end for c in chunks), line
    travel = [c for c in chunks if c.heading == "Travel Policy"]
    assert len(travel) >= 2  # topic shift (expenses -> laptops / table) splits the section


def test_short_document_is_one_chunk_and_long_prose_is_bounded():
    assert len(chunk_text("Start Date: December 14, 2026\nZIP: 80202\n")) == 1
    random.seed(7)
    words = "salary equity benefits start laptop address the of is".split()
    text = "\n\n".join(" ".join(random.choice(words) for _ in range(random.randint(10, 300))) + "." for _ in range(25))
    chunks = chunk_text(text)
    _invariants(text, chunks)
    assert max(len(c.text) for c in chunks) <= 1000


# ----------------------------------------------------------------- HNSW
def test_hnsw_filtered_search_widens_until_enough_hits():
    e = HashingEmbedder()
    ix = HNSWIndex(e.dim)
    for i in range(300):
        ix.add(f"d{i}", e.embed([f"start date december {i}"])[0], {"cls": "rare" if i % 100 == 0 else "common"})
    hits = ix.search(e.embed(["start date december"])[0], k=3, where=lambda p: p["cls"] == "rare")
    assert sorted(i for i, _, _ in hits) == ["d0", "d100", "d200"]  # all of them, despite HNSW misses
    assert not ix.add("d0", e.embed(["x"])[0], {})  # idempotent


async def test_long_documents_are_chunked_and_retrieved(app):
    from conftest import RECRUITER
    await ingest(app, "H1", hire(), "e1")
    filler = "\n\n".join(f"Section {i}. Policy text about travel, expenses and code of conduct." * 3 for i in range(60))
    text = f"Offer\n\n{filler}\n\nStart Date: December 14, 2026\n\n{filler}"
    r = await app.documents.upload("H1", "long.txt", text.encode(), "text/plain", RECRUITER, "offer_letter")
    assert r["chunks"] > 10
    ctx = app.documents.extraction_context(app.store.get_document(r["doc_hash"])["text_redacted"], r["doc_hash"])
    assert "Start Date: December 14, 2026" in ctx and len(ctx) < len(text) / 3


async def test_vector_index_rebuilds_from_db_with_acl_metadata(tmp_path):
    """Regression: rebuilt chunk payloads lost doc_type/classification, so filtered
    search denied everything after a restart."""
    from conftest import IT, PEOPLE_OPS, RECRUITER, make_app
    from hiresync import tools as toolreg
    db = str(tmp_path / "v.db")
    a = make_app(db)
    await a.start()
    await ingest(a, "H1", hire(), "e1")
    await a.documents.upload("H1", "o.txt", b"Start Date: December 14, 2026\n", "text/plain", RECRUITER, "offer_letter")
    n = len(a.vectors.chunks)
    await a.stop()
    await a.documents.upload("H1", "it.txt", b"IT Hardware Form\nLaptop: MacBook Pro 16\n", "text/plain",
                             PEOPLE_OPS, "it_hardware_form")
    n = len(a.vectors.chunks)
    await a.stop()
    b = make_app(db)
    await b.start()
    assert len(b.vectors.chunks) == n > 0
    out, err = await toolreg.execute(b, "search_documents", {"query": "laptop"}, principal=IT)
    hits = json.loads(out)
    assert not err and [h["doc_type"] for h in hits] == ["it_hardware_form"]
    await b.stop()
