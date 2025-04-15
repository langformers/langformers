[![PyPI](https://img.shields.io/pypi/v/langformers.svg)](https://pypi.org/project/langformers/) [![Downloads](https://static.pepy.tech/badge/langformers)](https://pepy.tech/project/langformers) [![Docs](https://img.shields.io/website?url=https%3A%2F%2Flangformers.com)](https://langformers.com) [![License](https://img.shields.io/github/license/langformers/langformers?color=blue)](https://github.com/langformers/langformers/blob/main/LICENSE)
  
# Langformers

[Langformers](https://langformers.com) is a flexible and user-friendly library that unifies NLP pipelines for both Large Language Models (LLMs) and Masked Language Models (MLMs) into one simple API.

**What makes Langformers special?**
Whether you're generating text, training classifiers, labelling data, embedding sentences, reranking sentences, or building a semantic search index... the API stays consistent:

```python
from langformers import tasks

component = tasks.create_<something>(...)
component.<do_something>()
```

No need to juggle different frameworks — Langformers brings Hugging Face Transformers, Ollama, FAISS, ChromaDB, Pinecone, and more under one unified interface.

Use the same pattern everywhere:

```python
tasks.create_generator(...)   # Chatting with LLMs
tasks.create_labeller(...)    # Data labelling using LLMs
tasks.create_embedder(...)    # Embeding Sentences
tasks.create_reranker(...)    # Reranking Sentences
tasks.create_classifier(...)  # Training a Text Classifier
tasks.create_tokenizer()      # Training a Custom Tokenizer
tasks.create_mlm(...)         # Pretraining an MLM
tasks.create_searcher(...)    # Vector Database search
tasks.create_mimicker(...)    # Knowledge Distillation
tasks.create_chunker(...)     # Chunking for LLMs
```

  
## Installation
Langformers can be installed using `pip`.

```bash
pip  install  -U  langformers
```

## Supported Tasks

Below are the pre-built NLP tasks available in Langformers. Each link points to an example in Langformer's documentation to help you get started quickly.

### Generative LLMs (e.g., Llama, Mistral, DeepSeek)

- Seamless Chat with LLMs
- LLM Inference via API
- Data Labelling with LLMs
- Chunking

### Masked Language Models (e.g., RoBERTa)

- Train Text Classifiers
- Pretrain MLMs from scratch
- Continue Pretraining MLMs on custom data

### Embeddings & Search (e.g., Sentence Transformers, FAISS, Pinecone)

- Embed Sentences
- Semantic Search
- Rerank Sentences
- Mimic a Pretrained Model (Knowledge Distillation)

## Documentation

Complete documentation and advanced usage examples are available at: https://langformers.com.

## License

Langformers is released under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).

## Contributing

We welcome contributions! Please see our contribution guidelines (at https://langformers.com/contributing.html) for details.

 ---
Built with ❤️ for the future of language AI.