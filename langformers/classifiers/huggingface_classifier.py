from dataclasses import dataclass, field
from transformers import TrainingArguments, EarlyStoppingCallback, AutoTokenizer, AutoModelForSequenceClassification
import os
from langformers.commons import device, get_name, print_message
from typing import Optional, Dict
from langformers.classifiers.customs import (CustomClassificationDataset, CustomClassificationModel,
                                             CustomClassificationTrainer)
from safetensors.torch import load_file
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import torch
from sklearn.metrics import f1_score


@dataclass
class TrainingConfig:
    """
    Default configuration for fine-tuning a Hugging Face model (e.g., BERT, RoBERTa, MPNet) on a text classification dataset. In addition to these parameters, you can pass the parameters ``class transformers.TrainingArguments`` takes. Refer: https://huggingface.co/docs/transformers/en/main_classes/trainer#transformers.TrainingArguments.

    Args:
        num_train_epochs (int, default=10): Number of epochs to train the model.
        save_total_limit (int, default=1): Limits the number of saved checkpoints to save disk space.
        learning_rate (float, default=2e-5): Learning rate. Controls the step size during optimization.
        per_device_train_batch_size (int, default=16): Batch size during training (per device).
        per_device_eval_batch_size (int, default=16): Batch size during evaluation (per device).
        save_strategy (str, default="steps"): When to save checkpoints during training.
        save_steps (int, default=100): Number of steps between model checkpoint saves.
        eval_strategy (str, default="steps"): When to run evaluation during training.
        logging_strategy (str, default="steps"): When to log training metrics ("steps", "epoch", etc.).
        logging_steps (int, default=100): Number of update steps between logging.
        report_to (list, default=["none"]): List of integrations to report to (e.g., "tensorboard", "wandb").
        logging_dir (str, default=None): Directory to save the training logs. If not provided, logging will be done in a timestamp-based directory inside logs/. (see run_name)
        output_dir (str, default=None): Directory to save model checkpoints. If not provided, logging will be done in a timestamp-based directory. (see run_name)
        run_name (str, default=<timestamp based string>): Descriptor for the run. Will be typically used by logging tools "wandb", "tensorboard" etc. Langformers will automatically generate a run name based on current timestamp.
        test_size (float, default=0.2): Proportion of data for test split.
        val_size (float, default=0.1): Proportion of data for validation split.
        metric_for_best_model (str, default="f1_macro"): Metric to use for comparing models.
        early_stopping_patience (int, default=5): Number of evaluations to wait before early stopping.
        early_stopping_threshold (float, default=0.0001): Minimum improvement threshold for early stopping.
        load_best_model_at_end (bool, default=True): Whether to load best model at end of training.
        max_length (int, default=None): Maximum sequence length for tokenization. If not provided, automatically assigned to tokenizer's `model_max_length`.
    """
    num_train_epochs: int = 10
    save_total_limit: int = 1
    learning_rate: float = 2e-5
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 16
    save_strategy: str = "steps"
    save_steps: int = 100
    eval_strategy: str = "steps"
    logging_strategy = "steps"
    logging_steps: int = 100
    report_to: list = field(default_factory=lambda: ["none"])
    logging_dir: str = None
    output_dir: str = None

    run_name: str = get_name("classifier")

    test_size: float = 0.2
    val_size: float = 0.1
    metric_for_best_model: str = 'f1_macro'
    early_stopping_patience: int = 5
    early_stopping_threshold = 0.0001
    load_best_model_at_end: bool = True

    max_length: int = None

    def __post_init__(self):
        self.logging_dir = f"./logs/{self.run_name}"
        self.output_dir = f"./{self.run_name}"


class ClassificationDataset:
    """
    Prepares the dataset in a way required for text classification task: performs data splits, encodes labels.
    """

    def __init__(self, csv_path, text_column="text", label_column="label"):
        try:
            self.df = pd.read_csv(csv_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"CSV file not found at path: {csv_path}")
        except pd.errors.EmptyDataError:
            raise ValueError(f"CSV file at {csv_path} is empty or invalid.")
        except Exception as e:
            raise ValueError(f"Error reading CSV file at {csv_path}: {e}")

        if text_column not in self.df.columns or label_column not in self.df.columns:
            raise ValueError(f"CSV file must contain columns '{text_column}' and '{label_column}'.")

        
        self.df.dropna(inplace=True)

        self.text_column = text_column
        self.label_column = label_column
        self.label_encoder = LabelEncoder()

    def split(self, test_size=0.2, val_size=0.1, random_state=42):
        """Split data into train/val/test."""

        train_val_df, test_df = train_test_split(
            self.df, test_size=test_size, random_state=random_state
        )

        train_df, val_df = train_test_split(
            train_val_df, test_size=val_size / (1 - test_size), random_state=random_state
        )

        print_message(f"Data splits: Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

        return train_df, val_df, test_df

    def encode_labels(self, train_df, val_df=None, test_df=None):
        """Fit label encoder on training data and transform all splits."""
        self.label_encoder.fit(train_df[self.label_column])

        encoded = {
            'train': self.label_encoder.transform(train_df[self.label_column]),
            'val': self.label_encoder.transform(val_df[self.label_column]) if val_df is not None else None,
            'test': self.label_encoder.transform(test_df[self.label_column]) if test_df is not None else None
        }

        return encoded


def get_class_weights(train_labels):
    """Compute class weights for imbalanced datasets"""
    classes = np.unique(train_labels)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=train_labels
    )
    return torch.tensor(weights, dtype=torch.float32)


def compute_metrics(pred):
    """Returns what evaluation metrics to be used during fine-tuning of an MLM."""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    return {"f1_macro": f1_score(labels, preds, average="macro"),
            "f1_weighted": f1_score(labels, preds, average="weighted"), }


class HuggingFaceClassifier:
    """
    Fine-tunes a masked language model (e.g., BERT, RoBERTa, MPNet) on a text classification task.

    This class handles dataset preparation, label encoding, model training, and evaluation.
    """

    def __init__(self, model_name: str, csv_path: str, text_column: str = "text", label_column: str = "label",
                 training_config: Optional[Dict] = None):
        """
        Initializes the fine-tuning process for a masked language model.

        Args:
            model_name (str, required): Name or path of the pretrained transformer model (e.g., "bert-base-uncased").
            csv_path (str, required): Path to the CSV file containing training data.
            text_column (str, default="text"): Column name in the CSV file containing the input text.
            label_column (str, default="label"): Column name in the CSV file containing labels.
            training_config (Optional[Dict], required): A dictionary containing training parameters. If not provided,
                                      default values will be assigned from :py:class:`langformers.classifiers.huggingface_classifier.TrainingConfig`.
        """
        self.model_name = model_name
        self.csv_path = csv_path
        self.text_column = text_column
        self.label_column = label_column
        self.device = device
        self.model = None
        self.tokenizer = None
        self.label_encoder = None

        training_config = training_config or {}

        self.training_config = TrainingConfig(
            **{k: v for k, v in training_config.items() if k in TrainingConfig.__annotations__})

        if isinstance(self.training_config.report_to, str):
            self.training_config.report_to = [self.training_config.report_to]

    def train(self):
        """
        Starts the fine-tuning process.
        """
        try:    
            dataset = ClassificationDataset(self.csv_path, self.text_column, self.label_column)

            train_df, val_df, test_df = dataset.split(test_size=self.training_config.test_size,
                                                    val_size=self.training_config.val_size)

            encoded = dataset.encode_labels(train_df, val_df, test_df)
            self.label_encoder = dataset.label_encoder
        except Exception as e:
            raise RuntimeError(f"Error during dataset preparation: {e}")
        
        try:
            id2label = {i: label for i, label in enumerate(self.label_encoder.classes_)}
            label2id = {label: i for i, label in id2label.items()}

            print_message(f"Found {len(id2label)} classes: {id2label}")

            self.model = CustomClassificationModel(
                model_name=self.model_name,
                num_labels=len(self.label_encoder.classes_)
            ).to(self.device)

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            print_message(f"Pretrained model and tokenizer loaded.")
        except Exception as e:
            raise RuntimeError(f"Error loading model or tokenizer: {e}")

        self.model.model.config.id2label = id2label
        self.model.model.config.label2id = label2id

        if self.training_config.max_length is None or self.training_config.max_length > self.tokenizer.model_max_length:
            print_message(
                f"Tokenization defaulting to max_length: {self.tokenizer.model_max_length}. max_length is either None or is >model_max_length.")
        try:
            train_dataset = self.create_dataset(train_df, encoded['train'])
            val_dataset = self.create_dataset(val_df, encoded['val'])
            test_dataset = self.create_dataset(test_df, encoded['test'])

            print_message(f"Train/Valid/Test Datasets created and loaded.")

            training_args_dict = vars(self.training_config).copy()
            exclude_keys = {"max_length", "test_size", "val_size", "early_stopping_threshold",
                            "early_stopping_patience"}
            filtered_args = {k: v for k, v in training_args_dict.items() if k not in exclude_keys}

            training_args = TrainingArguments(
                **filtered_args,
            )

            class_weights = get_class_weights(encoded['train']).to(device)
            print_message(f"Class weights computed. {class_weights}")

            trainer = CustomClassificationTrainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                compute_metrics=compute_metrics,
                class_weights=class_weights,
                callbacks=[EarlyStoppingCallback(self.training_config.early_stopping_patience,
                                                self.training_config.early_stopping_threshold)]
            )

            print_message(f"Training started.")

            trainer.train()
        except Exception as e:
            raise RuntimeError(f"Error during training: {e}")

        try:
            global latest_checkpoint_path
            best_model_checkpoint = trainer.state.best_model_checkpoint

            if best_model_checkpoint:
                model_weights = load_file(os.path.join(best_model_checkpoint, "model.safetensors"))
                self.model.load_state_dict(model_weights)

                best_model_path = os.path.join(training_args.output_dir, "best_model")
                os.makedirs(best_model_path, exist_ok=True)

                self.model.save_pretrained(best_model_path)
                self.tokenizer.save_pretrained(best_model_path)

                print_message(f"Best model saved at {best_model_path}")
            else:
                print_message("No valid best model checkpoint found.")

        except Exception as e:
            print_message(f"Error while saving/loading the model: {e}")

        try:
            print_message(f"Evaluating the saved model on Test set.")
            eval_results = trainer.evaluate(eval_dataset=test_dataset)
            print_message("------------------------------------")
            print_message("Evaluation Summary:")
            print_message("------------------------------------")
            print_message(f"{eval_results}")
        except Exception as e:
            print_message(f"Error during evaluation: {e}")

    def create_dataset(self, df, labels):
        """Performs tokenization on a dataset."""
        try:
            encodings = self.tokenizer(df[self.text_column].tolist(),
                                    truncation=True,
                                    padding="max_length",
                                    max_length=self.training_config.max_length,
                                    return_tensors="pt")

        except Exception as e:
            raise RuntimeError(f"Error during dataset preparation: {e}")
        
        return CustomClassificationDataset(encodings, labels)
