Langformers Documentation
===========================

🚀 **Unified NLP Pipelines for Language Models**

Langformers is a powerful yet user-friendly Python library designed for seamless interaction with large language models (LLMs) and masked language models (MLMs).

It unifies the following core NLP pipelines into a single, cohesive API:

- Conversational AI (:doc:`chat interface <chat>` and :doc:`REST api <llm-inference>`)
- :doc:`MLM pretraining <pretrain-mlms>`
- :doc:`Text classification <train-text-classifiers>`
- :doc:`Sentence embedding <embed-sentences>`
- :doc:`Reranking sentences <reranker>`
- :doc:`Data labelling <data-labelling-llms>`
- :doc:`Semantic search <semantic-search>`
- :doc:`Knowledge distillation <mimick-a-model>`

Langformers is built on top of popular libraries such as Pytorch\ [#]_, Transformers\ [#]_, Ollama\ [#]_,  FastAPI\ [#]_, ensuring compatibility with modern NLP workflows. The library supports Hugging Face and Ollama models, and is optimized for both CUDA and Apple Silicon (MPS).

.. admonition:: Installing
    :class: warning

    You can install **Langformers** using `pip`:

    .. code-block:: bash

       pip install -U langformers

Whether you're generating text, training classifiers, labelling data, embedding sentences, or building a semantic search index... the API stays consistent:

.. code-block:: python

    from langformers import tasks

    component = tasks.create_<something>(...)
    component.<do_something>()

No need to juggle different frameworks — Langformers wraps Hugging Face Transformers, SentenceTransformers, Ollama, FAISS, ChromaDB, Pinecone, and more under one unified interface.

Use the same pattern everywhere:

.. code-block:: python

    tasks.create_generator(...)  # Chatting with LLMs
    tasks.create_labeller(...)   # Data labelling using LLMs
    tasks.create_embedder(...)   # Embeding Sentences
    tasks.create_reranker(...)   # Reranking Sentences
    tasks.create_classifier(...) # Training a Text Classifier
    tasks.create_tokenizer()     # Training a Custom Tokenizer
    tasks.create_mlm(...)        # Pretraining an MLM
    tasks.create_searcher(...)   # Vector Database search
    tasks.create_mimicker(...)   # Knowledge Distillation


Tasks in Langformers
----------------------
Langformers delivers a smooth and unified experience for researchers and developers alike, supporting a broad set of essential NLP tasks right out of the box.

Below are the pre-built NLP tasks available:


.. image:: ./_static/tasks.svg
    :alt: Langformers Tasks
    :width: 100%
    :class: non-clickable


.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   dependencies

.. toctree::
   :maxdepth: 2
   :caption: LLMs

   chat
   llm-inference
   data-labelling-llms

.. toctree::
   :maxdepth: 2
   :caption: MLMs

   train-text-classifiers
   pretrain-mlms
   further-pretrain-mlms

.. toctree::
   :maxdepth: 2
   :caption: Embeddings

   embed-sentences
   semantic-search
   reranker
   mimick-a-model


.. toctree::
   :maxdepth: 1
   :caption: Library Reference

   api

.. toctree::
   :maxdepth: 1
   :caption: Misc

   license
   contributing
   changelog

**Footnotes**

.. [#] Pytorch: https://pytorch.org/docs/stable/index.html
.. [#] Transformers: https://huggingface.co/docs/transformers/en/index
.. [#] Ollama: https://ollama.com/search
.. [#] FastPI: https://fastapi.tiangolo.com

