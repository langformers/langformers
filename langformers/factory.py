from typing import Optional, Dict, Any, Callable
from .classifiers import LoadClassifier, HuggingFaceClassifier
from .embedders import HuggingFaceEmbedder
from .mlms import HuggingFaceMLMCreator, MLMTokenizerDatasetCreator
from .generators import OllamaGenerator, HuggingFaceGenerator
from .labellers import OllamaDataLabeller, HuggingFaceDataLabeller
from .mimickers import EmbeddingMimicker
from .searchers import FaissSearcher, ChromaDBSearcher, PineconeSearcher


class tasks:
    """
    Factory class for creating and managing various NLP tasks involving LLMs and MLMs.
    """
    @staticmethod
    def create_generator(provider: str, model_name: str, memory: bool = True, dependency: Optional[Callable[..., Any]] = None,
                         device: str = None):
        """
        Factory method for creating and managing LLMs chatting (User Interface) and LLM Inference (REST api).

        Args:
            provider (str, required): The model provider (e.g., “ollama”). Currently supported providers: ``ollama``, ``huggingface``.
            model_name (str, required): The model name from the provider’s hub (e.g., “llama3.1:8b”).
            memory (bool, default= True): Whether to save previous chat interactions to maintain context. Chatting with an LLM
                (via a user interface) definitely makes sense to maintain memory, which is why this option defaults to True.
                But, when doing LLM Inference (via REST api) there might be some use cases where maintaining contexts might not be useful.
                Therefore, this option exists.
            dependency (Optional[Callable[..., Any]], default=<no auth>): A FastAPI dependency.
                The callable can return any value which will be injected into the route `/api/generate`.
            device (str, default=None): The device to load the model, data on ("cuda", "mps" or "cpu").
                If not provided, device will automatically be inferred. Currently used for HuggingFace models, as input ids and attention mask need to be on the save device as the model.

        Returns:
            An instance of the appropriate generator class, based on the selected provider.

                - If `provider` is "huggingface", an instance of `HuggingFaceGenerator` is returned.
                - If `provider` is "ollama", an instance of `OllamaGenerator` is returned.
        """
        providers = {
            "huggingface": HuggingFaceGenerator,
            "ollama": OllamaGenerator
        }
        if provider not in providers:
            raise ValueError(f"Unsupported provider: {provider}. Available providers: {list(providers.keys())}")

        return providers[provider](model_name, memory=memory, device=device, dependency=dependency)

    @staticmethod
    def load_classifier(model_name: str):
        """
        Factory method for loading a custom classifier from disk.

        Args:
            model_name (str, required): Path to the classifier.

        Returns:
            An instance of `LoadClassifier`.
        """
        return LoadClassifier(model_name)

    @staticmethod
    def create_classifier(model_name: str, csv_path: str, text_column: str = "text", label_column: str = "label",
                          training_config: Optional[Dict] = None):
        """
        Factory method for training text classifiers. Only Hugging Face models are supported. Used for finetuning models such as BERT, RoBERTa, MPNet on a text classification dataset.

        Args:
            model_name (str, required): Name or path of the pretrained transformer model (e.g., "bert-base-uncased").
            csv_path (str, required): Path to the CSV file containing training data.
            text_column (str, default="text"): Column name in the CSV file containing the input text.
            label_column (str, default="label"): Column name in the CSV file containing labels.
            training_config (Optional[Dict], required): A dictionary containing training parameters. If not provided,
                                      default values will be assigned from :py:class:`langformers.classifiers.huggingface_classifier.TrainingConfig`.

        Returns:
            An instance of `HuggingFaceClassifier`.
        """
        return HuggingFaceClassifier(model_name, csv_path, text_column,
                                             label_column, training_config)

    @staticmethod
    def create_tokenizer(data_path: str, tokenizer_config: Optional[Dict] = None, tokenizer: str = None):
        """
        Factory method for training a tokenizer on a custom dataset, and use the trained tokenizer to tokenize the dataset.

        Args:
            data_path (str, required): Path to a raw text data (e.g., data.txt). Each line in the dataset should
                contain a single sentence or document. Each line can also be multiple sentences, but note that truncation
                will be applied.
            tokenizer_config (dict, default=None): Configurations for the tokenizer. If not provided,
                                      default values will be assigned from the dataclass `langformers.mlms.mlm_tokenizer_dataset_creator.TokenizerConfig`.
            tokenizer (str, default=None): Path to a trained tokenizer, such as “roberta-base” on Hugging Face, or a local path.
                If tokenizer is provided, it ignores `tokenizer_config`.

        Returns:
            An instance of `MLMDatasetCreator`.
        """
        return MLMTokenizerDatasetCreator(data_path, tokenizer_config, tokenizer)

    @staticmethod
    def create_mlm(tokenizer: str, tokenized_dataset: str, model_config: Optional[Dict] = None, training_config: Optional[Dict] = None, checkpoint_path=None):
        """
        Factory method for training a masked language model based on RoBERTa pretraining procedure.

        Args:
            tokenizer (str, required): Path to the trained tokenizer.
            tokenized_dataset (str, required): Path to the tokenized dataset.
            model_config (Optional[Dict], default=None): Dictionary containing model configurations.
                If None, the model must be loaded from a checkpoint.
            training_config (Optional[Dict], default=None): Dictionary containing training configurations.
                If None, default training configurations will be used.
            checkpoint_path (Optional[str], default=None): Path to a model checkpoint for resuming training.

        Returns:
            An instance of `HuggingFaceMLMCreator`.
        """
        return HuggingFaceMLMCreator(tokenizer, tokenized_dataset, model_config, training_config, checkpoint_path)

    @staticmethod
    def create_mimicker(teacher_model: str, student_config: dict, training_config: dict):
        """
        Factory method for creating smaller models by mimicking the vector space of a large teacher model.

        Args:
            teacher_model (str, required): Name of the teacher model from Hugging Face.
            student_config (dict, required): Configuration for the student model.
            training_config (dict, required): Configuration for training.

        Returns:
            An instance of `EmbeddingMimicker`.
        """
        return EmbeddingMimicker(teacher_model, student_config, training_config)

    @staticmethod
    def create_embedder(provider: str, model_name: str, **kwargs):
        """
        Factory method for creating a sentence embedder.

        Args:
            provider (str, required): The model provider (e.g., “huggingface”). Currently supported providers: ``huggingface``.
            model_name (str, required): The model name from the provider’s hub (e.g., "sentence-transformers/all-mpnet-base-v2").
            **kwargs (dict, required): Provider specific keyword arguments. Keeping this as more providers will be added.

        Returns:
            An instance of the appropriate embedder class, based on the selected provider.

                - If `provider` is "huggingface", an instance of `HuggingFaceEmbedder` is returned.
        """
        providers = {
            "huggingface": HuggingFaceEmbedder,
        }
        if provider not in providers:
            raise ValueError(f"Unsupported provider: {provider}. Available providers: {list(providers.keys())}")
        return providers[provider](model_name, **kwargs)

    @staticmethod
    def create_searcher(database: str, **kwargs):
        """
        Factory method for creating a searcher and perform vector search.

        Args:
            database (str, required): Type of vector database (e.g., "faiss"). Supported database: ``faiss``,
                ``chromadb``, and ``pinecone``.
            **kwargs (dict, required): Database specific keyword arguments.

        Returns:
            An instance of the appropriate searcher class, based on the selected database.

                - If `database` is "faiss", an instance of `FaissSearcher` is returned.
                - If `database` is "chromadb", an instance of `ChromaDBSearcher` is returned.
                - If `database` is "pinecone", an instance of `PineconeSearcher` is returned.
        """
        databases = {
            "faiss": FaissSearcher,
            "chromadb": ChromaDBSearcher,
            "pinecone": PineconeSearcher
        }

        if database not in databases:
            raise ValueError(f"Unsupported database: {database}. Available databases: {list(databases.keys())}")
        return databases[database](**kwargs)

    @staticmethod
    def create_labeller(provider: str, model_name: str, multi_label: bool = False):
        """
        Factory method for loading an LLM for data labelling tasks.

        Args:
            provider (str, required): The name of the embedding model provider (e.g., “huggingface”). Currently supported providers: ``huggingface``, ``ollama``.
            model_name (str, required): The model name from the provider’s hub (e.g., “llama3.1:8b” if provider is "Ollama").
            multi_label (bool, default=False): If True, allows multiple labels to be selected.

        Returns:
            An instance of the appropriate labeller class, based on the selected provider.

            - If `provider` is "ollama", an instance of `OllamaDataLabeler` will be returned.
            - If `provider` is "huggingface", an instance of `HuggingFaceDataLabeler` will be returned.
        """
        providers = {
            "ollama": OllamaDataLabeller,
            "huggingface": HuggingFaceDataLabeller
        }
        return providers[provider](model_name, multi_label)
