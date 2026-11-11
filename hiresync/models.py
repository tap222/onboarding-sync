"""Canonical hire record: the one shape every system is projected from.

Ashby (or Workday, for off-cycle hires) writes *changes* to this record. Downstream
systems never read each other; they only receive projections of this record.
"""
from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, EmailStr, Field


class Address(BaseModel):
    model_config = ConfigDict(extra="forbid")
    line1: str
    city: str
    state: Optional[str] = None
    postal_code: str
    country: str = Field(min_length=2, max_length=2, description="ISO-3166 alpha-2")


class WorkLocation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    country: str = Field(min_length=2, max_length=2)
    city: Optional[str] = None
    remote: bool = True


class HireRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    hire_id: str
    legal_first_name: str
    legal_last_name: str
    preferred_first_name: Optional[str] = None
    personal_email: EmailStr
    start_date: date
    job_title: str
    department: str
    manager_email: EmailStr
    employment_type: Literal["FTE", "contractor"] = "FTE"
    work_location: WorkLocation
    shipping_address: Address


def canonicalize(record: dict) -> dict:
    """Validated, defaults filled, JSON-typed. Every stored version goes through this,
    so projections can rely on every field being present."""
    return HireRecord.model_validate(record).model_dump(mode="json")


class Source(str, Enum):
    ASHBY = "ashby"
    WORKDAY = "workday"  # off-cycle hires created outside Ashby
    HUMAN = "human"  # an approver's correction
    SYSTEM = "system"


class Severity(str, Enum):
    BLOCK = "block"  # never propagate without a human
    WARN = "warn"  # propagate if triage is confident, else ask a human
    INFO = "info"


class Issue(BaseModel):
    code: str
    severity: Severity
    message: str
    field: Optional[str] = None
    systems: list[str] = []


class RunStatus(str, Enum):
    RUNNING = "running"
    AWAITING_APPROVAL = "awaiting_approval"
    COMPLETED = "completed"
    PARTIAL = "partial"  # some systems failed or were deferred; retryable
    REJECTED = "rejected"
    SUPERSEDED = "superseded"  # a newer version arrived; that run owns propagation
    COMPENSATED = "compensated"


class Step(str, Enum):
    START = "start"
    VALIDATED = "validated"
    TRIAGED = "triaged"
    APPROVED = "approved"
    PROPAGATED = "propagated"
    RECONCILED = "reconciled"
    DONE = "done"
    COMPENSATING = "compensating"
