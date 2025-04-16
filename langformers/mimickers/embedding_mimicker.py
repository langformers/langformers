from dataclasses import dataclass
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, RobertaModel, RobertaConfig
import torch.optim as optim
from tqdm import tqdm
from langformers.commons.device import device
import os
from langformers.commons import mean_pooling, print_message, get_name
from langformers.mimickers.customs import CustomMimickerDataset
from typing import Optional, Union



@dataclass
class StudentConfig:
    """
    Configuration parameters for the student model.

    Args:
        max_position_embeddings (int, required): Maximum sequence length for input to the student model.
        num_attention_heads (int, required): Number of attention heads in the student model.
        num_hidden_layers (int, required): Number of transformer layers in the student model.
        hidden_size (int, required): Size of the hidden layer.
        intermediate_size (int, required): Size of the intermediate layer.
    """
    max_position_embeddings: int
    num_attention_heads: int
    num_hidden_layers: int
    hidden_size: int
    intermediate_size: int


@dataclass
class TrainingConfig:
    """
    Default configuration for training.

    Args:
        num_train_epochs (int, required): The number of epochs to train the student model.
        learning_rate (float, required): The learning rate used in optimization.
        batch_size (int, required): The batch size used during training.
        dataset_path (list or str, required): List of sentences or path to a text corpus. The text corpus should have one sentence (or a few) per line. Anything beyond the max_length of the teacher’s tokenizer will be truncated.
        logging_steps (int, required): The number of steps between each log message during training.
    """
    num_train_epochs: int
    learning_rate: float
    batch_size: int
    dataset_path: Union[list, str]
    logging_steps: int


class EmbeddingMimicker:
    """
    Mimics the teacher model's vector space using a student model.

    This class is used to train a student model to replicate the embedding space of a pretrained
    teacher model. The student model learns to match the output embeddings of the teacher model by
    minimizing the mean squared error (MSE) loss between the teacher's and student's embeddings.
    """
    def __init__(self, teacher_model: str, student_config: Optional[dict] = None,
                 training_config: Optional[dict] = None):
        """
        Loads the teacher model and initializes the student model with provided configurations. Prepares dataset and dataloader required for the training.

        Args:
            teacher_model (str, required): Name of the teacher model on Hugging Face.
            student_config (dict, required): Configuration for the student model. Refer to :py:class:`langformers.mimickers.embedding_mimicker.StudentConfig` for key-value arguments.
            training_config (dict, required): Configuration for training. Refer to :py:class:`langformers.mimickers.embedding_mimicker.TrainingConfig` for key-value arguments.
        """
        self.teacher_model_name = teacher_model
        self.init_student_config = StudentConfig(**student_config)

        try:
            self.teacher_model = AutoModel.from_pretrained(self.teacher_model_name).to(device)
            self.tokenizer = AutoTokenizer.from_pretrained(self.teacher_model_name)
            print_message(f"Teacher model {self.teacher_model_name} and its tokenizer loaded.")
        except Exception as e:
            raise ValueError(f"Failed to load the teacher model '{self.teacher_model_name}': {e}")

        try:
            vocab_size = self.teacher_model.config.vocab_size
            
            setattr(self.init_student_config, 'vocab_size', vocab_size)
            self.student_config = RobertaConfig(**self.init_student_config.__dict__)
        except Exception as e:
            raise ValueError(f"Error initializing student configuration: {e}")
        
        try:
            self.training_config = TrainingConfig(**training_config)
        except Exception as e:
            raise ValueError(f"Error initializing training configuration: {e}")
        
        self.student_model = RobertaModel(config=self.student_config).to(device)
        print_message("Student model is initialized.")

        self.downsampler = Downsampler(input_dim=self.teacher_model.config.hidden_size,
                                       output_dim=self.student_config.hidden_size).to(device)

        try:
            self.max_length_for_tokenization = self.init_student_config.max_position_embeddings - 2
            self.dataset = CustomMimickerDataset(dataset_path=self.training_config.dataset_path, tokenizer=self.tokenizer,
                                                max_length=self.max_length_for_tokenization)

            print_message("Dataset loaded.")

            self.dataloader = DataLoader(self.dataset, batch_size=self.training_config.batch_size, shuffle=True)
        except Exception as e:
            raise RuntimeError(f"Error preparing dataset or dataloader: {e}")

        self.optimizer = optim.AdamW(
            list(self.student_model.parameters()) + list(self.downsampler.parameters()),
            lr=self.training_config.learning_rate
        )

    def train(self):
        """
        Starts the training.
        """
        try:
            train_model(
                self.student_model, self.teacher_model, self.tokenizer, self.downsampler, self.dataloader, self.optimizer,
                epochs=self.training_config.num_train_epochs, log_steps=self.training_config.logging_steps, max_length_for_tokenization=self.max_length_for_tokenization)

            print_message("Training finished. Final model saved in the 'best_model' directory.")
        except Exception as e:
            raise RuntimeError(f"Error during training: {e}")


class Downsampler(nn.Module):
    """Maps teacher's embeddings to student's."""
    def __init__(self, input_dim, output_dim):
        super(Downsampler, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)


def train_model(student_model, teacher_model, tokenizer, downsampler, dataloader, optimizer, epochs, log_steps, max_length_for_tokenization):
    """
    Trains the student model to mimic the teacher model's embeddings.

    This function trains the student model by minimizing the mean squared error (MSE) loss between
    the teacher's and student's embeddings. It also saves the model when the loss improves.

    Notes:
        - At the end of the training, saves the best performing model in the 'best_model' directory.
    """
    run_name = get_name("mimicker")
    os.makedirs(run_name, exist_ok=True)
    print_message(f"Training has started. At every logging step, if the training loss improves, a checkpoint will be saved in '{run_name}/best_model'.")

    tokenizer.model_max_length = max_length_for_tokenization
    tokenizer.save_pretrained(f"{run_name}/best_model")
    
    mse_loss = nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

    student_model.train()
    teacher_model.eval()
    downsampler.train()

    best_loss = float('inf')

    for epoch in range(epochs):
        epoch_loss = 0
        try:
            for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)

                with torch.no_grad():
                    teacher_outputs = teacher_model(input_ids=input_ids, attention_mask=attention_mask)
                    teacher_embeddings = mean_pooling(teacher_outputs, attention_mask)
                    teacher_embedding_down_sampled = downsampler(teacher_embeddings)

                if device.type == 'cuda':
                    with torch.cuda.amp.autocast():
                        student_outputs = student_model(input_ids=input_ids, attention_mask=attention_mask)
                        student_embedding = mean_pooling(student_outputs, attention_mask)
                        loss = mse_loss(student_embedding, teacher_embedding_down_sampled)
                else:
                    student_outputs = student_model(input_ids=input_ids, attention_mask=attention_mask)
                    student_embedding = mean_pooling(student_outputs, attention_mask)
                    loss = mse_loss(student_embedding, teacher_embedding_down_sampled)

                optimizer.zero_grad()
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                epoch_loss += loss.item()

                if (batch_idx + 1) % log_steps == 0:
                    avg_loss = epoch_loss / (batch_idx + 1)

                    if avg_loss < best_loss:
                        best_loss = avg_loss                        
                        student_model.save_pretrained(f"{run_name}/best_model")
                        print_message(f"Epoch: {epoch + 1}, Training steps: {batch_idx + 1}, Loss: {best_loss:.4f}")
        except Exception as e:
            raise RuntimeError(f"Error during epoch {epoch + 1}: {e}")
