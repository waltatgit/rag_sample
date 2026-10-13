"""Retriever module for fetching relevant study evidence given criteria."""

from typing import List
from llama_index.core import VectorStoreIndex
from modules.models import EvidencePassage
from modules.utils import setup_logger

logger = setup_logger()


class StudyRetriever:
    """Performs semantic similarity retrieval on a specified LlamaIndex study index."""

    def __init__(self, index: VectorStoreIndex, study_label: str, top_k: int = 5):
        self.retriever = index.as_retriever(similarity_top_k=top_k)
        self.study_label = study_label

    def retrieve_evidence(self, criterion: str) -> List[EvidencePassage]:
        """Retrieves similarity-matched passages from the study's vector index."""
        nodes = self.retriever.retrieve(criterion)
        passages: List[EvidencePassage] = []

        for node in nodes:
            metadata = node.metadata
            filename = metadata.get("filename", "Unknown")
            score = float(node.score) if node.score is not None else 0.0

            passages.append(
                EvidencePassage(
                    text=node.get_content(),
                    filename=filename,
                    relevance_score=score,
                    study=self.study_label
                )
            )

        logger.info(f"Retrieved {len(passages)} passages for '{self.study_label}' on criterion: '{criterion[:30]}...'")
        return passages