from transformers import AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from langformers.chunkers import FixedSizeChunker
from langformers.embedders import HuggingFaceEmbedder
from langformers.commons import print_message


class SemanticChunker:
    def __init__(self, model_name: str):
        """
        Initializes the SemanticChunker class.
        
        Args:
            model_name (str): The name of the Hugging Face model to use for embedding. The model's tokenizer will be used for tokenization.
        """
        
        try:
            self.embedder = HuggingFaceEmbedder(model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        except Exception as e:
            raise ValueError(f"Error loading model: {e}")
        
        self.first_chunker = FixedSizeChunker(model_name)

    
    def chunk(self, document: str, initial_chunk_size: int, max_chunk_size: int, similarity_threshold: float = 0.2):
        """
        Chunk the document into semantic segments based on cosine similarity.
        
        Args:
            document (str, required): The document to be chunked. If the document is something like PDF, it should be converted to a string first.
            initial_chunk_size (int, required): The maximum size of chunks to be created for merging later.
            max_chunk_size (int, required): The maximum size of the final chunks.
            similarity_threshold (float, default=0.2): The threshold for cosine similarity to merge chunks.
        """

        chunks = self.first_chunker.chunk(document, chunk_size=initial_chunk_size)
        embeddings = self.embedder.embed(chunks)
        
        final_chunks = []

        temp_chunk = ""
        temp_chunk_size = 0

        for i, chunk in enumerate(chunks):
            tokenized_chunk = self.tokenizer.encode(chunk, add_special_tokens=False)

            if i == 0:
                temp_chunk = chunk
                temp_chunk_size = len(tokenized_chunk)
                continue

            similarity = cosine_similarity(embeddings[i - 1].reshape(1, -1), embeddings[i].reshape(1, -1))[0][0]

            if temp_chunk_size + len(tokenized_chunk) <= max_chunk_size and similarity >= similarity_threshold:
                temp_chunk += " " + chunk
                temp_chunk_size += len(tokenized_chunk)
            else:
                final_chunks.append(temp_chunk)
                temp_chunk = chunk
                temp_chunk_size = len(tokenized_chunk)

        if temp_chunk:
            final_chunks.append(temp_chunk)

        return final_chunks