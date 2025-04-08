import torch
from dataclasses import dataclass, field
from transformers import DataCollatorForLanguageModeling, RobertaForMaskedLM, Trainer, TrainingArguments
from transformers import get_linear_schedule_with_warmup
import torch.optim as optim
from transformers import RobertaConfig, RobertaTokenizer
from typing import Optional, Dict
from langformers.commons import get_name, print_message
from datasets import load_from_disk


@dataclass
class ModelConfig:
    """
    Default configuration for the model. Architecture will default to roberta-base architecture, if None. In addition to these parameters, you can pass the parameters class ``transformers.RobertaConfig`` takes. Refer: https://huggingface.co/docs/transformers/en/model_doc/roberta#transformers.RobertaConfig.

    Args:
        vocab_size (int, default=50_265): Size of the vocabulary (must match with the tokenizer).
        max_position_embeddings (int, default=512): Maximum sequence length the model can handle.
        num_attention_heads (int, default=12): Number of attention heads in the Transformer.
        num_hidden_layers (int, default=12): Number of layers in the Transformer encoder.
        hidden_size (int, default=768): Dimensionality of the hidden layers.
        intermediate_size (int, default=3072): Dimensionality of the feedforward layer in the Transformer.
        attention_probs_dropout_prob (float, default=0.1): Dropout probability for attention layers to prevent overfitting.
    """
    model_config: Dict = field(default_factory=dict)


@dataclass
class TrainingConfig:
    """
    Default configuration for pretraining an MLM with RoBERTa pretraining procedure. In addition to these parameters, you can pass the parameters class ``transformers.TrainingArguments`` takes. Refer: https://huggingface.co/docs/transformers/en/main_classes/trainer#transformers.TrainingArguments.

    Args:
        num_train_epochs (int, default=10): Number of epochs to train the model.
        save_total_limit (int, default=10): Limits the number of saved checkpoints to save disk space.
        learning_rate (float, default=0.0002): Learning rate. Controls the step size during optimization.
        per_device_train_batch_size (int, default=32): Batch size during training (per device).
        gradient_accumulation_steps (int, default=2): Simulates a larger batch size by accumulating gradients from multiple batches before performing a weight update.
        save_strategy (str, default="steps"): When to save checkpoints during training.
        save_steps (int, default=100): Number of steps between model checkpoint saves.
        logging_steps (int, default=100): Number of update steps between logging.
        report_to (list, default=[“none”]): List of integrations to report to (e.g., "tensorboard", "wandb").
        mlm_probability (float, default=0.15): Probability of masking tokens during masked language modeling (MLM).
        warmup_ratio (float, default=0.05): Fraction of total training steps used for learning rate warmup.
        logging_dir (str, default=None): Directory to save the training logs. If not provided, logging will be done in a timestamp-based directory inside logs/. (see run_name)
        output_dir (str, default=None): Directory to save model checkpoints. If not provided, logging will be done in a timestamp-based directory. (see run_name)
        n_gpus (int, default=1): Number of GPUs to train with. Automatically computed for CUDA. Used for computing total steps of the training.
        run_name (str, default=<timestamp based string>): Descriptor for the run. Will be typically used by logging tools "wandb", "tensorboard" etc. Langformers will automatically generate a run name based on current timestamp.
    """
    num_train_epochs: int = 10
    save_total_limit: int = 10
    learning_rate: float = 0.0002
    per_device_train_batch_size: int = 32
    gradient_accumulation_steps: int = 2
    save_strategy: str = "steps"
    save_steps: int = 100
    logging_steps: int = 100
    report_to: list = field(default_factory=lambda: ["none"])
    mlm_probability: float = 0.15
    warmup_ratio: float = 0.05
    logging_dir: str = None
    output_dir: str = None
    n_gpus: int = field(default_factory=lambda: torch.cuda.device_count() if torch.cuda.is_available() else 1)

    run_name: str = get_name("mlm")

    def __post_init__(self):
        self.logging_dir = f"./logs/{self.run_name}"
        self.output_dir = f"./{self.run_name}"


class HuggingFaceMLMCreator:
    """
    Trains a masked language model (MLM) using the RoBERTa pretraining procedure.

    This class sets up and trains a RoBERTa-based masked language model using a tokenized dataset.
    It supports loading from a checkpoint or initializing a new model with a given configuration.
    """
    def __init__(self, tokenizer: str, tokenized_dataset: str, model_config: Optional[Dict] = None,
                 training_config: Optional[Dict] = None, checkpoint_path: Optional[str] = None):
        """
        Initializes the masked language model (MLM) training process.

        Args:
            tokenizer (str, required): Path to the tokenizer.
            tokenized_dataset (str, required): Path to the tokenized dataset.
            model_config (Optional[Dict], default=None): Dictionary containing model configurations.
                If None, `checkpoint_path` must be provided a valid checkpoint path.
            training_config (Optional[Dict], default=None): Dictionary containing training configurations.
                If None, default values will be assigned from :py:class:`langformers.mlms.mlm_trainer.TrainingConfig`.
            checkpoint_path (Optional[str], default=None): Path to a model checkpoint for resuming training.
        """
        self.tokenizer = RobertaTokenizer.from_pretrained(tokenizer)
        
        try:
            self.tokenized_dataset = load_from_disk(tokenized_dataset)
        except Exception as e:
            raise ValueError(f"Error loading dataset from {tokenized_dataset}. Ensure the path is correct.")

        training_config = training_config or {}
        extra_args = {k: v for k, v in training_config.items() if k not in TrainingConfig.__annotations__}

        self.training_config = TrainingConfig(
            **{k: v for k, v in training_config.items() if k in TrainingConfig.__annotations__})
        self.training_config.extra_args = extra_args

        if checkpoint_path:
            print_message(f"Loading model from checkpoint: {checkpoint_path}")
            self.model = RobertaForMaskedLM.from_pretrained(checkpoint_path)
        elif model_config:
            try:
                print_message("Initializing a new model with provided model_config.")
                self.model = RobertaForMaskedLM(config=RobertaConfig(**(model_config or {})))
            except Exception as e:
                raise ValueError("Error initializing model with provided model_config. Ensure the config is valid.")
        else:
            raise ValueError("Either `model_config` or `checkpoint_path` must be provided.")

    def train(self):
        """
        Starts the training process.

        The final model is saved at:
        - `{self.training_config.output_dir}/final_model` (default location).

        The logs are saved at:
        - `{self.training_config.output_dir}/training_logs` (default location).

        """
        try:
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer, mlm=True, mlm_probability=self.training_config.mlm_probability
            )

            total_steps = max(1, (len(self.tokenized_dataset['train']) // (
                    self.training_config.per_device_train_batch_size * self.training_config.gradient_accumulation_steps
                    * self.training_config.n_gpus) + 1) * self.training_config.num_train_epochs)

            warmup_steps = max(1, int(total_steps * self.training_config.warmup_ratio))

            training_args_dict = vars(self.training_config).copy()
            exclude_keys = {"n_gpus", "mlm_probability", "warmup_ratio", "extra_args"}
            filtered_args = {k: v for k, v in training_args_dict.items() if k not in exclude_keys}

            training_args = TrainingArguments(
                warmup_steps=warmup_steps,
                **filtered_args,
                **self.training_config.extra_args
            )

            optimizer = optim.AdamW(self.model.parameters(), lr=self.training_config.learning_rate)

            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
            )

            trainer = Trainer(
                model=self.model,
                args=training_args,
                data_collator=data_collator,
                train_dataset=self.tokenized_dataset['train'],
                optimizers=(optimizer, scheduler),
            )

            print_message(f"Training started. Model checkpoints will be saved inside {self.training_config.output_dir}.")

            if hasattr(self, "checkpoint_path"):
                print_message(f"Resuming training from checkpoint: {self.checkpoint_path}")
                trainer.train(resume_from_checkpoint=self.checkpoint_path)
            else:
                trainer.train()

            print_message("Training finished...")

            trainer.save_model(f"{self.training_config.output_dir}/final_model")
            print_message(f"Final model saved at {self.training_config.output_dir}/final_model.")
        except Exception as e:
            raise RuntimeError("An error occurred during training.")
