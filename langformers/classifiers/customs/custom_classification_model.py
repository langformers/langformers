import torch
from transformers import AutoModel, AutoConfig
from langformers.commons import mean_pooling
from langformers.commons.device import device
import os


class CustomClassificationModel(torch.nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(self.model.config.hidden_size, num_labels)
        self.device = device

        self.model_type = self.model.config.model_type

    def forward(self, input_ids, attention_mask, labels=None):
        model_output = self.model(input_ids=input_ids, attention_mask=attention_mask)

        embeddings = mean_pooling(model_output, attention_mask)

        logits = self.classifier(self.dropout(embeddings))

        loss = None
        if labels is not None:
            loss_fn = torch.nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)

        return {"loss": loss, "logits": logits} if loss is not None else {"logits": logits}

    def save_pretrained(self, save_directory):
        """Save the model and the classifier head."""
        self.model.save_pretrained(save_directory)
        torch.save(self.classifier.state_dict(), os.path.join(save_directory, "classifier.bin"))

    @classmethod
    def from_pretrained(cls, model_path):
        """Load the custom model from a directory."""
        config = AutoConfig.from_pretrained(model_path)

        model = AutoModel.from_pretrained(model_path, config=config)

        classifier_path = os.path.join(model_path, "classifier.bin")
        classifier_state_dict = torch.load(classifier_path, weights_only=True)

        num_labels = classifier_state_dict["weight"].shape[0]

        custom_model = cls(model_path, num_labels)
        custom_model.model = model
        custom_model.classifier.load_state_dict(classifier_state_dict)

        return custom_model
