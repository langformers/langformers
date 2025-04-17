from typing import Optional, List, Any, Dict
import os
import sqlite3
import numpy as np
from langformers.commons import get_name, print_message
from transformers import AutoConfig
from langformers.embedders import HuggingFaceEmbedder
import json
import importlib


class FaissSearcher:
    """
    A FAISS-based semantic search engine for vectorized text retrieval.

    This class allows efficient nearest neighbor search using FAISS and stores text data in an SQLite database.
    It supports different FAISS index types (`FLAT` and `HNSW`) and provides methods for adding, querying,
    and managing the search index.
    """

    def __init__(self, embedder: str = None, index_path: Optional[str] = None,
                 db_path: Optional[str] = None, index_type: Optional[str] = "HNSW",
                 index_parameters: Optional[dict] = None):
        """
        Initializes the FAISS searcher.

        Args:
            embedder (str, default=None): Name of the HuggingFace embedding model.
            index_path (Optional[str], default=None): Path to the FAISS index file. If None, a new index is created.
            db_path (Optional[str], default=None): Path to the SQLite database file. If None, a new database is created.
            index_type (str, default="HNSW"): Type of FAISS index. Supported types are ``FLAT`` and ``HNSW``.
            index_parameters (Optional[dict], default=None): Additional parameters for the FAISS index. If not provided, the following is used.

                .. code-block:: python

                    index_parameters = { "m": 32, "efConstruction": 40 }
        """
        try:
            self.faiss = importlib.import_module("faiss")
        except ImportError:
            raise ImportError("FAISS is not installed. Please install it using 'pip install langformers[faiss]'.")

        self.index = None
        if embedder is None:
            raise ValueError("An embedding model must be provided.")

        self.model_name = embedder
        self.config = AutoConfig.from_pretrained(self.model_name)
        self.dimension = self.config.hidden_size
        self.index_type = index_type.upper()

        self.index_parameters = {
            "m": 32,
            "efConstruction": 40
        }

        if index_parameters:
            self.index_parameters.update(index_parameters)

        self.output_dir = get_name("faiss")

        if not db_path and not index_path:
            os.makedirs(self.output_dir, exist_ok=True)
            print_message(f"No existing database and index provided. Both will be created inside '{self.output_dir}'.")

        self.index_file = index_path or os.path.join(self.output_dir, "faiss.index")
        self.db_file = db_path or os.path.join(self.output_dir, "faiss.db")

        self.init_db()
        self.load_or_create_index()
        self.embedder = HuggingFaceEmbedder(model_name=self.model_name)

    def init_db(self):
        """Initializes the SQLite database for storing text data."""
        self.conn = sqlite3.connect(self.db_file)
        self.cursor = self.conn.cursor()

        self.cursor.execute("""
                    CREATE TABLE IF NOT EXISTS texts (
                        id INTEGER PRIMARY KEY,
                        text TEXT,
                        metadata TEXT
                    )
                """)

        self.conn.commit()

    def load_or_create_index(self):
        """Loads an existing FAISS index from file or creates a new one."""
        if os.path.exists(self.index_file):
            self.index = self.faiss.read_index(self.index_file)
            print_message(f"Loaded existing FAISS index '{self.index_file}' with '{self.index_type}' search from '{self.output_dir}'.")
        else:
            self.create_faiss_index()
            print_message(f"Created new FAISS index '{self.index_file}' with '{self.index_type}' search.")

    def create_faiss_index(self):
        """Creates a new FAISS index based on configuration."""
        if self.index_type == "FLAT":
            self.index = self.IndexFlatL2(self.dimension)
        elif self.index_type == "HNSW":
            self.index = self.faiss.IndexHNSWFlat(
                self.dimension,
                self.index_parameters["m"]
            )
            self.index.hnsw.efConstruction = self.index_parameters["efConstruction"]
        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")

    def train_index(self, embeddings):
        """Trains the index if it requires training."""
        if hasattr(self.index, "is_trained") and not self.index.is_trained:
            print_message("Training FAISS index...")
            self.index.train(embeddings)
            print_message("FAISS index trained.")

    def add(self, texts: List[str], metadata: Optional[List[Dict[str, Any]]] = None):
        """
        Adds text data to the FAISS index and stores it in the SQLite database.

        Args:
            texts (List[str]): List of text entries to be indexed.
            metadata (Optional[List[dict]], default=None): Metadata associated with each text.

        Notes:
            - The texts are first converted to embeddings.
            - The embeddings are then added to the FAISS index.
            - The text data is stored in the SQLite database for retrieval.
        """
        if metadata is not None and len(metadata) != len(texts):
            raise ValueError("Texts and metadata must be the same length.")

        try:
            embeddings = self.embedder.embed(texts=texts)
            self.train_index(embeddings)

            start_id = self.index.ntotal
            db_entries = []

            for i, text in enumerate(texts):
                meta = metadata[i] if metadata else None
                meta_json = json.dumps(meta) if meta else None
                db_entries.append((start_id + i, text, meta_json))

            self.cursor.executemany(
                "INSERT INTO texts (id, text, metadata) VALUES (?, ?, ?)",
                db_entries
            )
            self.index.add(embeddings)
            self.conn.commit()
            self.save_index()

            if metadata is not None:
                print_message(f"Added {len(texts)} new texts, embeddings and metadata to the FAISS index.")
            else:
                print_message(f"Added {len(texts)} new texts to the FAISS index.")
        except Exception as e:
            raise RuntimeError(f"Failed to add data to the index: {str(e)}")

    def query(self, query: str, items: int = 1, include_metadata: bool = True):
        """
        Queries the FAISS index to find similar texts.

        Args:
            query (str, required): The input text query.
            items (int, default=1): Number of nearest neighbors to retrieve.
            include_metadata (bool, default=True): Whether to include the metadata in the results.

        Notes:
            - The query text is first converted into an embedding.
            - The similarity score is calculated as `1 - (distance / 2)`, where distance is the FAISS L2 distance.
        """
        
        try:
            items = max(1, min(items, self.index.ntotal))
            query_embedding = self.embedder.embed(texts=[query])
            query_embedding = np.array(query_embedding, dtype=np.float32)

            distances, indices = self.index.search(query_embedding, items)
            results = []

            for i, idx in enumerate(indices[0]):
                item = self.get_by_id(idx)

                text = item["text"]
                metadata = item["metadata"]

                if include_metadata:
                    results.append({
                        "text": text,
                        "similarity": float(1 - (distances[0][i] / 2)) if distances[0][i] else 0,
                        "metadata": metadata
                    })
                else:
                    results.append({
                        "text": text,
                        "similarity": float(1 - (distances[0][i] / 2)) if distances[0][i] else 0,
                    })

            return results
        except Exception as e:
            raise RuntimeError(f"Failed to query the FAISS index: {str(e)}")

    def get_by_id(self, idx: int) -> Optional[Dict[str, Any]]:
        """Retrieves a complete record by ID."""
        try:
            self.cursor.execute(
                "SELECT text, metadata FROM texts WHERE id = ?",
                (int(idx),))
            row = self.cursor.fetchone()

            if not row:
                return None

            text, meta_json = row

            return {
                "text": text,
                "metadata": json.loads(meta_json) if meta_json else {}
            }
        except sqlite3.Error as e:
            raise RuntimeError(f"Failed to retrieve data from SQLite: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Failed to retrieve data: {str(e)}")

    def save_index(self):
        """Saves the FAISS index to disk."""
        self.faiss.write_index(self.index, self.index_file)

    def count(self):
        """
        Returns the number of items in the FAISS index.
        """
        return self.index.ntotal

    def close(self):
        """Closes the SQLite database connection."""
        self.conn.close()
