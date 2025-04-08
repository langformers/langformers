from typing import Optional, List, Any, Dict
from langformers.commons import get_name, print_message
from transformers import AutoConfig
from langformers.embedders import HuggingFaceEmbedder
import uuid
import os
from pinecone import Pinecone, ServerlessSpec


class PineconeSearcher:
    """
    A Pinecone-based semantic search engine for vectorized text retrieval.

    This class integrates with Pinecone to store and search for text embeddings,
    using a Hugging Face embedding model for vectorization.
    """

    def __init__(self, embedder: str = None, index_name: Optional[str] = None,
                 api_key: str = None, index_parameters: Optional[dict] = None):
        """
        Initializes the Pinecone searcher.

        Args:
            embedder (str): Name of the Hugging Face embedding model.
            index_name (Optional[str], default=None): Name of the Pinecone index.
            api_key (str): Pinecone API key for authentication.
            index_parameters (Optional[dict], default=None): Additional parameters for index configuration. If not provided, the following is used.

                .. code-block:: python

                    index_parameters = {'cloud': 'aws',
                                        'region': 'us-east-1',
                                        'metric': 'cosine',
                                        'dimension': self.dimension  # embedding model's output dimension.
                                        }
        """
        self.index = None
        if embedder is None:
            raise ValueError("An embedding model must be provided.")

        self.api_key = api_key or os.getenv("PINECONE_API_KEY")
        if self.api_key is None:
            raise ValueError(
                "Pinecone API key must be provided either directly "
                "or through PINECONE_API_KEY environment variable"
            )

        self.pc = Pinecone(api_key=self.api_key)

        self.model_name = embedder
        self.config = AutoConfig.from_pretrained(self.model_name)
        self.dimension = self.config.hidden_size

        self.index_parameters = {
            'cloud': 'aws',
            'region': 'us-east-1',
            'metric': 'cosine',
            'dimension': self.dimension
        }

        if index_parameters:
            self.index_parameters.update(index_parameters)

        self.index_name = index_name or get_name("pinecone")
        self.initialize_index()
        self.embedder = HuggingFaceEmbedder(model_name=self.model_name)

    def initialize_index(self):
        try:
            existing_indexes = self.pc.list_indexes()
            index_names = [index.name for index in existing_indexes] if hasattr(existing_indexes,
                                                                                'names') else existing_indexes

            if self.index_name not in index_names:
                self.pc.create_index(
                    name=self.index_name,
                    dimension=self.index_parameters['dimension'],
                    metric=self.index_parameters['metric'],
                    spec=ServerlessSpec(
                        cloud=self.index_parameters['cloud'],
                        region=self.index_parameters['region']
                    )
                )
                print_message(f"Created new Pinecone index: {self.index_name}")
            else:
                print_message(f"Using existing Pinecone index: {self.index_name}")

            self.index = self.pc.Index(self.index_name)

        except Exception as e:
            raise RuntimeError(f"Failed to initialize Pinecone index: {str(e)}")

    def add(self, texts: List[str], metadata: Optional[List[Dict[str, Any]]] = None):
        """
        Adds text data to the Pinecone index.

        Args:
            texts (List[str]): List of text entries to be indexed.
            metadata (Optional[List[dict]], default=None): Metadata associated with each text.

        Notes:
            - Each text is converted into an embedding.
            - A unique UUID is generated for each text entry.
            - The embeddings, along with metadata, are upserted into the Pinecone index.
        """
        if metadata is not None and len(metadata) != len(texts):
            raise ValueError("Texts and metadata must be the same length.")

        try:
            embeddings = [embedding.tolist() for embedding in self.embedder.embed(texts=texts)]
            ids = [str(uuid.uuid4()) for _ in texts]

            vectors = []
            for i, (embedding, text) in enumerate(zip(embeddings, texts)):
                meta = {"text": text}
                if metadata and metadata[i]:
                    meta.update(metadata[i])

                vectors.append((ids[i], embedding, meta))

            self.index.upsert(vectors=vectors)

            if metadata is not None:
                print_message(f"Added {len(texts)} new texts, embeddings and metadata to the Pinecone index.")
            else:
                print_message(f"Added {len(texts)} new texts to the Pinecone index.")
        except Exception as e:
            raise RuntimeError(f"Failed to add data to the Pinecone index:: {str(e)}")

    def query(self, query: str, items: int = 1, include_metadata: bool = True):
        """
        Queries the Pinecone index to find similar texts.

        Args:
            query (str, required): The input text query.
            items (int, default=1): Number of nearest neighbors to retrieve.
            include_metadata (bool, default=True): Whether to include the metadata in the results.
        """
        try:
            items = max(1, items)
            query_embedding = self.embedder.embed(texts=[query])[0].tolist()

            results = self.index.query(
                vector=query_embedding,
                top_k=items,
                include_metadata=include_metadata,
            )

            if include_metadata:
                return [{
                    "text": match.metadata["text"],
                    "similarity": match.score,
                    "id": match.id,
                    "metadata": match.metadata,
                } for match in results.matches]
            else:
                return [{
                    "id": match.id,
                    "similarity": match.score,
                } for match in results.matches]
        except Exception as e:
            raise RuntimeError(f"Failed to query the Pinecone index: {str(e)}")

    def count(self):
        """Returns the number of items stored in the Pinecone index."""
        stats = self.index.describe_index_stats()
        return stats["total_vector_count"]
