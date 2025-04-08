from transformers import Trainer
import torch
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class CustomClassificationTrainer(Trainer):
    """Custom trainer with class weights"""

    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights if class_weights is not None else None

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get('logits')
        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(logits.view(-1, self.model.model.config.num_labels), labels.view(-1))

        return (loss, outputs) if return_outputs else loss
