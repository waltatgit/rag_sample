"""Manages creation, chunking, and local persistence of independent LlamaIndex vector stores."""

import os
from typing import List
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document
from modules.utils import setup_logger

logger = setup_logger()


class IndexManager:
    """Handles index generation, persistence, and loading for a study folder."""

    def __init__(self, storage_dir: str, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.storage_dir = storage_dir
        self.node_parser = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def get_or_create_index(self, documents: List[Document]) -> VectorStoreIndex:
        """Loads index from local storage if existing; creates and saves it otherwise."""
        if os.path.exists(self.storage_dir) and os.listdir(self.storage_dir):
            logger.info(f"Loading existing persisted index from {self.storage_dir}")
            storage_context = StorageContext.from_defaults(persist_dir=self.storage_dir)
            index = load_index_from_storage(storage_context)
        else:
            logger.info(f"Creating new VectorStoreIndex for {self.storage_dir}")
            os.makedirs(self.storage_dir, exist_ok=True)
            
            # Chunking documents into nodes
            nodes = self.node_parser.get_nodes_from_documents(documents)
            index = VectorStoreIndex(nodes)
            
            # Local persistence via SimpleVectorStore (LlamaIndex default)
            index.storage_context.persist(persist_dir=self.storage_dir)
            logger.info(f"Persisted index to {self.storage_dir}")

        return index