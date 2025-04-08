from typing import Any
import torch
from transformers import AutoTokenizer, AutoConfig
from langformers.classifiers.customs import CustomClassificationModel
from langformers.commons.device import device
import torch.nn.functional as F


class LoadClassifier:
    """
        Loads a text classification model trained with Langformers.

        This class loads a custom classification model, tokenizes input text, and predicts
        labels with their associated probabilities.
        """
    def __init__(self, model_name: str):
        """
        Loads the provided custom classification model. Retrieves label mappings if available in the model configuration.

        Args:
            model_name (str, required): Path to the classifier.

        """
        self.model = CustomClassificationModel.from_pretrained(model_name).to(device)
        self.config = AutoConfig.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.label2id = self.config.label2id if hasattr(self.config, 'label2id') else {}
        self.id2label = self.config.id2label if hasattr(self.config, 'id2label') else {}

    def classify(self, texts: list[str]) -> list[dict[str, Any]]:
        """
        Classifies input texts into predefined categories.

        Args:
            texts (list[str], required): A list of text strings to classify.

        Notes:
            - Tokenizes input texts and feeds them into the model.
            - Uses softmax to obtain probability distributions.
            - Returns the most probable class label for each input text.
        """
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

            logits = outputs["logits"]

            probabilities = F.softmax(logits, dim=-1)

            predicted_classes = torch.argmax(probabilities, dim=-1).tolist()

            result = []
            for i in range(len(texts)):
                label = self.id2label.get(predicted_classes[i], str(predicted_classes[i]))
                label_probability = probabilities[i][predicted_classes[i]].item()
                result.append({
                    "label": label,
                    "prob": label_probability
                })

            return result
