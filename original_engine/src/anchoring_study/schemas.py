from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ClinicalResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    most_likely_diagnosis: str
    ranked_differential_diagnoses: list[str] = Field(min_length=3, max_length=5)
    triage_level: Literal["emergency", "same_day", "urgent_outpatient", "routine"]
    recommended_next_steps: list[str] = Field(min_length=2, max_length=8)
    brief_rationale: str

