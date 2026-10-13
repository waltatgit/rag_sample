"""Utility functions for file handling, logging, and JSON cleanup."""

import os
import json
import logging
import re
from typing import List, Dict, Any

# Configure application logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/app.log",
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("study_assessor")


def setup_logger() -> logging.Logger:
    """Returns application logger instance."""
    return logger


def parse_criteria_file(filepath: str) -> List[str]:
    """Reads a text file containing criteria (one per line or numbered) and returns a list."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Criteria file not found: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    criteria = []
    for line in lines:
        cleaned = line.strip()
        if cleaned:
            # Strip leading numbering like "1. ", "Criterion 1:", "- "
            cleaned = re.sub(r"^(\d+[\.\)]|[-*]|Criterion\s*\d+:?)\s*", "", cleaned, flags=re.IGNORECASE)
            if cleaned:
                criteria.append(cleaned)
    return criteria


def extract_json_from_response(raw_text: str) -> Dict[str, Any]:
    """Extracts and parses JSON from markdown code fences or raw string LLM outputs."""
    pattern = r"```(?:json)?\s*(\{.*?\})\s*```"
    match = re.search(pattern, raw_text, re.DOTALL)
    
    if match:
        clean_str = match.group(1)
    else:
        clean_str = raw_text.strip()

    try:
        return json.loads(clean_str)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON string: {raw_text}")
        raise ValueError(f"Invalid JSON returned from LLM response: {e}")