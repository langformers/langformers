from typing import Optional, List, Any, Dict
from transformers import AutoConfig
from langformers.embedders import HuggingFaceEmbedder
import uuid
from langformers.commons import get_name, print_message
import chromadb


class ChromaDBSearcher:
    """
        A ChromaDB-based semantic search engine for vectorized text retrieval.

        This class integrates with ChromaDB to store and search for text embeddings,
        using a Hugging Face embedding model for vectorization.
    """
    def __init__(self, embedder: str = None, db_path: Optional[str] = None,
                 collection_name: Optional[str] = "my_collection"):
        """
        Initializes the ChromaDB searcher.

        Args:
            embedder (str): Name of the Hugging Face embedding model.
            db_path (Optional[str], default=None): Path to the ChromaDB database. If None, a default name is generated.
            collection_name (Optional[str], default="my_collection"): Name of the ChromaDB collection.
        """
        self.collection = None
        if embedder is None:
            raise ValueError("An embedding model must be provided.")

        self.model_name = embedder
        self.config = AutoConfig.from_pretrained(self.model_name)
        self.dimension = self.config.hidden_size
        self.db_path = db_path or get_name("chromadb")
        self.collection_name = collection_name or None
        self.client = chromadb.PersistentClient(self.db_path)
        self.embedder = HuggingFaceEmbedder(model_name=self.model_name)

        self.initialize_collection()

    def initialize_collection(self):
        try:
            self.collection = self.client.get_collection(name=self.collection_name)
            print_message(f"Loaded existing collection '{self.collection_name}' from '{self.db_path}'")
        except Exception as e:
            if "does not exist" in str(e) or isinstance(e, ValueError):
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"embedder": self.model_name,
                              "hnsw:space": "cosine"}
                )
                print_message(f"Created new collection '{self.collection_name}' inside '{self.db_path}'")
            else:
                raise RuntimeError(f"Failed to initialize collection: {str(e)}")

    def add(self, texts: List[str], metadata: Optional[List[Dict[str, Any]]] = None):
        """
        Adds text data to the ChromaDB collection.

        Args:
            texts (List[str], required): List of text entries to be indexed.
            metadata (Optional[List[dict]], default=None): Metadata associated with each text.

        Notes:
            - Each text is converted into an embedding.
            - A unique UUID is generated for each text entry.
            - The embeddings, along with metadata, are added to the ChromaDB collection.
        """
        if metadata is not None and len(metadata) != len(texts):
            raise ValueError("Texts and metadata must be the same length.")

        try:
            embeddings = [embedding.tolist() for embedding in self.embedder.embed(texts=texts)]

            ids = [str(uuid.uuid4()) for _ in range(len(texts))]

            metadata = metadata if metadata is not None else [None] * len(texts)

            self.collection.add(
                documents=texts,
                embeddings=embeddings,
                ids=ids,
                metadatas=metadata
            )

            if metadata is not None:
                print_message(f"Added {len(texts)} new texts, embeddings and metadata to the ChromaDB collection.")
            else:
                print_message(f"Added {len(texts)} new texts to the ChromaDB collection.")
        except Exception as e:
            raise RuntimeError(f"Failed to add data to the collection: {str(e)}")

    def query(self, query: str, items: int = 1, include_metadata: bool = True):
        """
        Queries the ChromaDB collection to find similar texts.

        Args:
            query (str, required): The input text query.
            items (int, default=1): Number of nearest neighbors to retrieve.
            include_metadata (bool, default=True): Whether to include the metadata in the results.

        Notes:
            - The query text is first converted into an embedding.
            - The similarity is derived from the squared distance returned by ChromaDB.
        """
        try:
            items = max(1, items)
            query_embedding = self.embedder.embed(texts=[query])
            query_embedding = query_embedding[0].tolist()

            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=items,
            )

            formatted_results = []
            for i in range(len(results["documents"][0])):
                result = {
                    "text": results["documents"][0][i],
                    "similarity": 1 - results["distances"][0][i]
                }
                if include_metadata and results.get("metadatas"):
                    result["metadata"] = results["metadatas"][0][i] or {}
                formatted_results.append(result)

            return formatted_results
        except Exception as e:
            raise RuntimeError(f"Failed to query the ChromaDB collection: {str(e)}")

    def count(self):
        """Return the number of items in the collection."""
        return self.collection.count()

    def close(self):
        """Close the ChromaDB client."""
        self.client.reset()
