Langformers Documentation
===========================

.. raw:: html
   :file: ./_includes/official_links.html

Langformers is a powerful yet user-friendly Python library designed for seamless interaction with large language models (LLMs) and masked language models (MLMs). It unifies the following core NLP pipelines into a single, cohesive API:

- Conversational AI (:doc:`chat interface <chat>` and :doc:`REST api <llm-inference>`)
- :doc:`MLM pretraining <pretrain-mlms>`
- :doc:`Text classification <train-text-classifiers>`
- :doc:`Sentence embedding <embed-sentences>`
- :doc:`Reranking sentences <reranker>`
- :doc:`Data labelling <data-labelling-llms>`
- :doc:`Semantic search <semantic-search>`
- :doc:`Knowledge distillation <mimick-a-model>`
- :doc:`Chunking for LLMs <chunking-for-llms>`

Langformers is built on top of popular libraries such as Pytorch\ [#]_, Transformers\ [#]_, Ollama\ [#]_,  FastAPI\ [#]_, ensuring compatibility with modern NLP workflows. The library supports Hugging Face and Ollama models, and is optimized for both CUDA and Apple Silicon (MPS).

.. admonition:: Installing
    :class: warning

    You can install **Langformers** using `pip`:

    .. code-block:: bash

       pip install -U langformers

    Requires **Python 3.10+**. For more details, check out the :doc:`installation guide <installation>`. 

.. admonition:: What makes Langformers special?
   :class: note

   Whether you're generating text, training classifiers, labelling data, embedding sentences, or building a semantic search index... the API stays consistent:

   .. code-block:: python

      from langformers import tasks

      component = tasks.create_<something>(...)
      component.<do_something>()

   No need to juggle different frameworks — Langformers brings Hugging Face Transformers, Ollama, FAISS, ChromaDB, Pinecone, and more under one unified interface.


Tasks in Langformers
----------------------
Langformers delivers a smooth and unified experience for researchers and developers alike, supporting a broad set of essential NLP tasks right out of the box.

Below are the pre-built NLP tasks available:


.. image:: ./_static/tasks.svg
    :alt: Langformers Tasks
    :width: 100%
    :class: non-clickable


Citing
--------
If you find Langformers useful in your research or projects, feel free to cite the following publication:


.. code-block:: bibtex

   @article{lamsal2025langformers,
      title={Langformers: Unified NLP Pipelines for Language Models}, 
      author={Rabindra Lamsal and Maria Rodriguez Read and Shanika Karunasekera},
      year={2025},
      journal={arXiv preprint arXiv:2504.09170},
      url={https://arxiv.org/abs/2504.09170}
   }


.. toctree::
   :maxdepth: 2
   :caption: Getting Started
   :hidden:

   installation
   dependencies

.. toctree::
   :maxdepth: 2
   :caption: LLMs
   :hidden:

   chat
   llm-inference
   data-labelling-llms
   chunking-for-llms

.. toctree::
   :maxdepth: 2
   :caption: MLMs
   :hidden:

   train-text-classifiers
   pretrain-mlms
   further-pretrain-mlms

.. toctree::
   :maxdepth: 2
   :caption: Embeddings
   :hidden:

   embed-sentences
   semantic-search
   reranker
   mimick-a-model


.. toctree::
   :maxdepth: 1
   :caption: Library Reference
   :hidden:

   api

.. toctree::
   :maxdepth: 1
   :caption: Development
   :hidden:

   changelog
   license
   contributing

**Footnotes**

.. [#] Pytorch: https://pytorch.org/docs/stable/index.html
.. [#] Transformers: https://huggingface.co/docs/transformers/en/index
.. [#] Ollama: https://ollama.com/search
.. [#] FastPI: https://fastapi.tiangolo.com

