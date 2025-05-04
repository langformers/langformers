from transformers import AutoTokenizer
from langformers.commons import print_message
from tqdm import tqdm


class FixedSizeChunker:
    """
    A class to chunk a document into fixed-size segments using a specified tokenizer. Also supports overlap between chunks.
    """
    def __init__(self, tokenizer: str):
        """
        Initializes the FixedSizeChunker class.
        
        Args:
            tokenizer (str, required): The tokenizer to use for encoding the document.
        """
        self.tokenizer_name = tokenizer

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
            print_message("Tokenizer loaded successfully.")
        except Exception as e:
            raise ValueError(f"Error loading tokenizer: {e}")

    def chunk(self, document: str, chunk_size: int = None, overlap: int = 0):
        """
        Splits the document into fixed-size chunks based on the tokenizer's encoding.

        Args:
            document (str, required): The document to be chunked. If the document is something like PDF, it should be converted to a string first.
            chunk_size (int, default=None): The size of each chunk. If not provided, the tokenizer's max length will be used.
            overlap (int, default=0): The number of tokens to overlap between consecutive chunks.
        """

        if chunk_size is None:
            chunk_size = self.tokenizer.model_max_length
            print_message("Chunk size is not provided. Using tokenizer's max length.")

        if overlap is None:
            overlap = 0
            print_message("Overlap is not provided. Using default value of 0.")

        if chunk_size <= overlap:
            raise ValueError("Chunk size must be greater than overlap size.")


        try:
            print_message("Encoding the document...")
            tokens = self.tokenizer.encode(document, add_special_tokens=False)
            print_message("Document encoded successfully.")
        except Exception as e:
            raise ValueError(f"Error encoding the document: {e}")

        chunked_tokens = []

        for i in range(0, len(tokens), chunk_size - overlap):
            chunked_tokens.append(self.tokenizer.decode(tokens[i:i + chunk_size], skip_special_tokens=True, clean_up_tokenization_spaces=True).strip())
        
        return chunked_tokens