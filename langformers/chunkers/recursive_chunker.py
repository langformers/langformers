from transformers import AutoTokenizer

class RecursiveChunker:
    def __init__(self, tokenizer: str):
        """
        Initializes the RecursiveChunker class.
        
        Args:
            tokenizer (str): The tokenizer to use for encoding the document.
        """
        self.tokenizer_name = tokenizer
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
            self.chunk_size = self.tokenizer.model_max_length
        except Exception as e:
            raise ValueError(f"Error loading tokenizer: {e}")
        
    def count_tokens(self, document):
        """Counts the number of tokens in a document."""
        return len(self.tokenizer.encode(document, add_special_tokens=False))
        
    def chunk(self, document: str, separators=["\n\n", "\n"]):

        """
        Chunk the document recursively based on the specified separators.
        
        Args:
            document (str, required): The document to be chunked. If the document is something like PDF, it should be converted to a string first.
            separators (list, default=["\n\n", "\n"]): The list of separators to use for chunking.
        """
        
        if not isinstance(document, str):
            raise ValueError("Document must be a string.")
        
        if not isinstance(separators, list) or not all(isinstance(s, str) for s in separators):
            raise ValueError("Separators must be a list of strings.")
        
        final_chunks = []
        separator = separators[0] if separators else None

        if self.count_tokens(document) > self.chunk_size:
            if separator:
                parts = document.split(separator)
                for part in parts:
                    if part.strip():
                        if self.count_tokens(part) > self.chunk_size:
                            final_chunks.extend(self.chunk(part.strip(), separators[1:]))
                        else:
                            final_chunks.append(part.strip())
            else:
                tokens = self.tokenizer.tokenize(document)
                for i in range(0, len(tokens), self.chunk_size):
                    chunk = self.tokenizer.convert_tokens_to_string(tokens[i:i + self.chunk_size])
                    final_chunks.append(chunk.strip())
        else:
            final_chunks.append(document.strip())

        return final_chunks