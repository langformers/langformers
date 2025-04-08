from transformers import AutoTokenizer, AutoModel
from langformers.commons import mean_pooling, print_message
import torch
from torch import nn
import torch.nn.functional as F


class HuggingFaceEmbedder:
    """
    Embeds sentences using Hugging Face models.

    This class generates embeddings for input texts and computes similarity scores between two given texts
    using cosine similarity.
    """
    def __init__(self, model_name: str):
        """
        Loads the embedding model and its tokenizer.

        Args:
            model_name (str, required): The model name from the provider’s hub (e.g., "sentence-transformers/all-mpnet-base-v2").
        """
        try:
            self.model = AutoModel.from_pretrained(model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        except Exception as e:
            raise RuntimeError(f"Failed to load model or tokenizer: {e}")

    def embed(self, texts: list[str]):
        """
        Generates normalized sentence embeddings for input texts.

        Args:
            texts (list[str]): A list of text sequences to be embedded.

        Notes:
            - Uses mean pooling over token embeddings to obtain sentence embeddings.
            - Applies L2 normalization to the embeddings.
        """
        if not isinstance(texts, list) or not all(isinstance(text, str) for text in texts):
            raise ValueError("Input must be a list of strings.")
        
        try:
            tokenized_input = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

            with torch.no_grad():
                model_output = self.model(**tokenized_input)

            mean_pooled_embeds = mean_pooling(model_output, tokenized_input['attention_mask'])

            normalized_embeds = F.normalize(mean_pooled_embeds, p=2, dim=1)

            return normalized_embeds

        except Exception as e:
            print_message(f"An error occurred during embedding: {e}")
            raise RuntimeError("Failed to generate embeddings.")

    def similarity(self, texts: list[str]):
        """
        Computes cosine similarity between two input texts.

        Args:
            texts (list): A list containing exactly two text sequences.

        Notes:
            - The similarity score ranges from -1 (completely different) to 1 (identical).
        """
        if not isinstance(texts, list) or len(texts) !=2 or not all(isinstance(text, str) for text in texts):
            raise ValueError("Input must be a list of exactly two strings.")
        
        try:
            embeds = self.embed(texts)
            return self.cos(embeds[0].unsqueeze(0), embeds[1].unsqueeze(0)).item()
        except Exception as e:
            raise RuntimeError("Failed to compute similarity.")