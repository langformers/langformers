import torch
from typing import List
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from langformers.commons import device, print_message

class CrossEncoder:
    """
    Ranks text pairs using a cross-encoder model from Hugging Face.
    """
    def __init__(self, model_name: str):
        """
        Loads the cross-encoder model and its tokenizer.

        Args:
            model_name (str, required): The model name from Hugging Face (e.g., "cross-encoder/ms-marco-MiniLM-L-6-v2")
        """
        self.model_name = model_name
        
        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            print_message(f"Loaded model and tokenizer for {model_name}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model or tokenizer for {model_name}: {e}")
        
    def rank(self, query: str, documents: List[str]):
        """
        Predict the ranking scores for the given pairs of texts. The function expects a query and a list of documents. Returns the documents sorted by their scores.
            
        Args:
            query (str, required): Query. E.g., "What is the capital of France?"
            documents (List(str), required): List of documents to be reranked. E.g., ["Paris is the capital of France.", "Berlin is the capital of Germany."]
        """     
        if not isinstance(query, str) and not isinstance(documents, List(str)):
            raise ValueError("Query must be a string and texts must be a list of strings.")
        
        final_query = [[query, document] for document in documents]

        inputs = self.tokenizer(final_query, return_tensors='pt', padding=True, truncation=True).to(device)
        
        try:
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(**inputs)
        except Exception as e:
            raise RuntimeError(f"Failed to encode provided text pairs: {e}")
        
        scores = outputs.logits.squeeze(-1).tolist()
        
        scored_documents = [{"score": score, "document": document, "original_index": index} for score, document, index in zip(scores, documents, range(len(documents)))]

        return sorted(scored_documents, key=lambda x: x["score"], reverse=True)
