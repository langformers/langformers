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
pip install -U langformers
```

This installs the latest version with [core dependencies](https://langformers.com/dependencies.html).

### Optional Dependencies

Langformers includes optional integrations you can install depending on your use case:

- For **FAISS** support: ``pip install -U langformers[faiss]``
- For **ChromaDB** support: ``pip install -U langformers[chromadb]``
- For **Pinecone** support: ``pip install -U langformers[pinecone]``

- To install **all optional features**: ``pip install -U langformers[all]``

## Supported Tasks

Below are the pre-built NLP tasks available in Langformers. Each link points to an example in Langformer's documentation to help you get started quickly.

### Generative LLMs (e.g., Llama, Mistral, DeepSeek)

- [Seamless Chat with LLMs](https://langformers.com/chat.html)
- [LLM Inference via API](https://langformers.com/llm-inference.html)
- [Data Labelling with LLMs](https://langformers.com/data-labelling-llms.html)
- [Chunking](https://langformers.com/chunking-for-llms.html)

### Masked Language Models (e.g., RoBERTa)

- [Train Text Classifiers](https://langformers.com/train-text-classifiers.html)
- [Pretrain MLMs from scratch](https://langformers.com/pretrain-mlms.html)
- [Continue Pretraining MLMs on custom data](https://langformers.com/further-pretrain-mlms.html)

### Embeddings & Search (e.g., Sentence Transformers, FAISS, Pinecone)

- [Embed Sentences](https://langformers.com/embed-sentences.html)
- [Semantic Search](https://langformers.com/semantic-search.html)
- [Rerank Sentences](https://langformers.com/reranker.html)
- [Mimic a Pretrained Model (Knowledge Distillation)](https://langformers.com/mimick-a-model.html)

## Documentation

Complete documentation and advanced usage examples are available at: https://langformers.com.

## License

Langformers is released under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).

## Contributing

We welcome contributions! Please see our [contribution guidelines](https://langformers.com/contributing.html) for details.

 ---
Built with ❤️ for the future of language AI.