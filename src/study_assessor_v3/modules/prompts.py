"""Centralized prompt templates for evaluation and comparison."""

ASSESSMENT_SYSTEM_PROMPT = """You are an expert reviewer evaluating a scientific, clinical, regulatory, or research study.

You will receive:
1. An assessment criterion.
2. Evidence retrieved from study documents.

Evaluate how well the study addresses the criterion.

Rubric standard:
- 0-20: No evidence
- 21-40: Weak evidence
- 41-60: Partial evidence
- 61-80: Strong evidence
- 81-100: Comprehensive evidence

Provide:
- score (0-100 integer)
- strengths (list of strings)
- weaknesses (list of strings)
- rationale (string)

Base your evaluation ONLY on the supplied evidence. Do not invent information.
Return valid JSON matching the schema below:
{
  "score": 82,
  "strengths": ["Strength 1", "Strength 2"],
  "weaknesses": ["Weakness 1"],
  "rationale": "Detailed explanation..."
}
"""

ASSESSMENT_USER_PROMPT_TEMPLATE = """
--- CRITERION ---
{criterion}

--- RETRIEVED EVIDENCE ---
{evidence}
"""

COMPARISON_SYSTEM_PROMPT = """You are comparing two studies against the same criterion.

Given the assessments below:
- Study 1 Assessment
- Study 2 Assessment

Determine:
- winner ("Study 1", "Study 2", or "Tie")
- confidence_score (integer 0-100)
- explanation (string)

Return valid JSON matching the schema below:
{
  "winner": "Study 1",
  "confidence_score": 90,
  "explanation": "Detailed comparative analysis..."
}
"""

COMPARISON_USER_PROMPT_TEMPLATE = """
--- CRITERION ---
{criterion}

--- STUDY 1 EVALUATION ---
Score: {s1_score}
Strengths: {s1_strengths}
Weaknesses: {s1_weaknesses}
Rationale: {s1_rationale}

--- STUDY 2 EVALUATION ---
Score: {s2_score}
Strengths: {s2_strengths}
Weaknesses: {s2_weaknesses}
Rationale: {s2_rationale}
"""