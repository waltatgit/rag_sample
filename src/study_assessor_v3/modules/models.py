"""Pydantic data models for structured outputs and evaluations."""

from typing import List, Optional
from pydantic import BaseModel, Field


class CriterionAssessment(BaseModel):
    """Assessment result for a single study against a specific criterion."""
    score: int = Field(
        ...,
        ge=0,
        le=100,
        description="Score between 0 and 100 based on rubric evidence strength."
    )
    strengths: List[str] = Field(
        default_factory=list,
        description="List of key strengths identified from evidence."
    )
    weaknesses: List[str] = Field(
        default_factory=list,
        description="List of key weaknesses or missing items identified from evidence."
    )
    rationale: str = Field(
        ...,
        description="Detailed evidence-backed explanation for the given score."
    )


class StudyComparison(BaseModel):
    """Comparison evaluation result between two study assessments."""
    winner: str = Field(
        ...,
        description="Study 1, Study 2, or Tie."
    )
    confidence_score: int = Field(
        ...,
        ge=0,
        le=100,
        description="Confidence score for the winning determination (0-100)."
    )
    explanation: str = Field(
        ...,
        description="Comparative breakdown explaining why one study performed better."
    )


class EvidencePassage(BaseModel):
    """Container for retrieved context passages."""
    text: str
    filename: str
    relevance_score: float
    study: str