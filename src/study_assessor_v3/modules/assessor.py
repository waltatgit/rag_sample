"""Module running LLM-driven evaluation and side-by-side comparison logic."""

from typing import List, Dict, Any, Tuple
from llama_index.core.llms import LLM
from modules.models import CriterionAssessment, StudyComparison, EvidencePassage
from modules.prompts import (
    ASSESSMENT_SYSTEM_PROMPT,
    ASSESSMENT_USER_PROMPT_TEMPLATE,
    COMPARISON_SYSTEM_PROMPT,
    COMPARISON_USER_PROMPT_TEMPLATE
)
from modules.utils import extract_json_from_response, setup_logger

logger = setup_logger()


class StudyAssessor:
    """Executes single study evaluations and cross-study comparisons using LLM."""

    def __init__(self, llm: LLM):
        self.llm = llm

    def evaluate_criterion(self, criterion: str, evidence: List[EvidencePassage]) -> CriterionAssessment:
        """Evaluates one study against a single criterion based strictly on provided evidence."""
        formatted_evidence = "\n\n".join([
            f"[Source File: {e.filename} | Relevance: {e.relevance_score:.2f}]\n{e.text}"
            for e in evidence
        ]) if evidence else "NO EVIDENCE FOUND."

        prompt = ASSESSMENT_USER_PROMPT_TEMPLATE.format(
            criterion=criterion,
            evidence=formatted_evidence
        )

        full_prompt = f"{ASSESSMENT_SYSTEM_PROMPT}\n\n{prompt}"
        response = self.llm.complete(full_prompt)
        
        json_data = extract_json_from_response(response.text)
        return CriterionAssessment(**json_data)

    def compare_assessments(
        self,
        criterion: str,
        study1_eval: CriterionAssessment,
        study2_eval: CriterionAssessment
    ) -> StudyComparison:
        """Compares two study evaluations for a given criterion."""
        prompt = COMPARISON_USER_PROMPT_TEMPLATE.format(
            criterion=criterion,
            s1_score=study1_eval.score,
            s1_strengths=", ".join(study1_eval.strengths),
            s1_weaknesses=", ".join(study1_eval.weaknesses),
            s1_rationale=study1_eval.rationale,
            s2_score=study2_eval.score,
            s2_strengths=", ".join(study2_eval.strengths),
            s2_weaknesses=", ".join(study2_eval.weaknesses),
            s2_rationale=study2_eval.rationale
        )

        full_prompt = f"{COMPARISON_SYSTEM_PROMPT}\n\n{prompt}"
        response = self.llm.complete(full_prompt)

        json_data = extract_json_from_response(response.text)
        return StudyComparison(**json_data)