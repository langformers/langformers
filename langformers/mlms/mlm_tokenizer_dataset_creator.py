from tokenizers.implementations import ByteLevelBPETokenizer
from dataclasses import dataclass, field
import os
from transformers import RobertaTokenizer
from datasets import load_dataset
from langformers.commons import print_message
from typing import Optional, Dict


@dataclass
class TokenizerConfig:
    """
    Configuration for the tokenizer.

    Args:
        max_length (int, default=512): Maximum sequence length for tokenization. For each line in the dataset, any sentence longer than this length will be truncated.
        vocab_size (int, default=50_265): Size of the vocabulary.
        min_frequency (int, default=2): Minimum frequency for a token to make its way into vocabulary.
        path_to_save_tokenizer (str, default="./tokenizer")
        special_tokens (List[str], default=["<s>","<pad>","</s>","<unk>","<mask>"]): Special tokens.
    """
    vocab_size: int = 50_265
    min_frequency: int = 2
    max_length: int = 512
    special_tokens: list[str] = field(default_factory=lambda: [
        "<s>",
        "<pad>",
        "</s>",
        "<unk>",
        "<mask>",
    ])
    path_to_save_tokenizer: str = "./tokenizer"


class MLMTokenizerDatasetCreator:
    """
    Trains a tokenizer on a custom dataset and tokenizes the dataset required for MLM training.
    """
    def __init__(self, data_path: str, tokenizer_config: Optional[Dict] = None, tokenizer: str = None):
        """
        Initializes the provided `data_path` and `tokenizer_config`.

        Args:
            data_path (str, required): Path to a raw text data (e.g., data.txt). Each line in the dataset should
                contain a single sentence or document. Each line can also be multiple sentences, but note that truncation
                will be applied.
            tokenizer_config (Optional[Dict], default=None): Configurations for the tokenizer. If not provided,
                                      default values will be assigned from :py:class:`langformers.mlms.mlm_tokenizer_dataset_creator.TokenizerConfig`.
            tokenizer (str, default=None): Path to a trained tokenizer, such as “roberta-base” on Hugging Face, or a local path.
                If tokenizer is provided, it ignores `tokenizer_config`.
        """
        self.tokenizer = None
        self.data_path = data_path
        self.tokenizer_path = tokenizer

        if not self.tokenizer_path:
            tokenizer_config = tokenizer_config or {}
            self.tokenizer_config = TokenizerConfig(**tokenizer_config)

    def train(self):
        """
        Loads a tokenizer if one is provided, otherwise trains a new one. After the tokenizer has been trained,
        it is persisted on the disk. The provided tokenizer or if one was trained, is loaded for tokenization.
        Tokenized dataset is then persisted on the disk.
        """
        if self.tokenizer_path:
            self.tokenizer = RobertaTokenizer.from_pretrained(self.tokenizer_path)
            print_message(f"Loaded tokenizer: {self.tokenizer_path}")
        else:
            print_message("Trained tokenizer not provided. Creating a new tokenizer from scratch.")
            tokenizer = ByteLevelBPETokenizer()

            if not os.path.exists(self.tokenizer_config.path_to_save_tokenizer):
                os.makedirs(self.tokenizer_config.path_to_save_tokenizer)

            print_message("Tokenizer training started.")
            tokenizer.train(files=self.data_path, vocab_size=self.tokenizer_config.vocab_size,
                            min_frequency=self.tokenizer_config.min_frequency,
                            special_tokens=self.tokenizer_config.special_tokens)

            tokenizer.save_model(self.tokenizer_config.path_to_save_tokenizer)
            print_message(f"Tokenizer has been trained and saved in: {self.tokenizer_config.path_to_save_tokenizer}.")

            self.tokenizer = RobertaTokenizer.from_pretrained(self.tokenizer_config.path_to_save_tokenizer)

        print_message("Tokenization started.")
        dataset = load_dataset("text", data_files=self.data_path)

        def tokenize_function(examples):
            try:
                return self.tokenizer(examples["text"],
                                    truncation=True,
                                    padding="max_length",
                                    max_length=self.tokenizer.model_max_length if self.tokenizer_path else self.tokenizer_config.max_length,
                                    return_special_tokens_mask=True,
                                    )
            except Exception as e:
                raise ValueError(f"Error during tokenization: {e}")

        tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
        tokenized_dataset.save_to_disk("tokenized_dataset")
        print_message("Tokenization finished. Tokenized data has been saved in: ./tokenized_dataset.")
    