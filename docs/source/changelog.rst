Changelog
===========

All notable changes to this project are documented here. The format is based on `Keep a Changelog <https://keepachangelog.com/en/1.0.0/>`_,
and this project adheres to `Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_.

v0.4.0 (2024-04-17)
---------------------

Changed
^^^^^^^^^
- Langformers now requires Python 3.10 or higher.
- Made the dependencies in the semantic search pipleines optional. Previously, FAISS, ChromaDB and Pinecone were required to install Langformers.
- Documentation updated.


v0.3.1 (2024-04-16)
---------------------

Changed
^^^^^^^
- Fixed a "CUDA error" in the mimicking pipeline which was caused due to incorrect use of tokenizer's max_length and model's max_position_embeddings. CPU and Apple Silicon devices did not have this issue.
- Documentation updated.

Added
^^^^^^
- Precision and Recall metrics (macro and weighted) added to classification pipleine.


v0.3.0 (2024-04-14)
---------------------

Added
^^^^^^
- Chunking pipelines
    - Fixed-size chunking
    - Semantic chunking
- Documentation and README.md updated.

v0.2.0 (2024-04-10)
---------------------

Added
^^^^^^
- Reranking pipeline.
- Documentation and README.md updated.

Changed
^^^^^^^
- Device (cuda, mps, cpu) check logic improved.


v0.1.0 (2024-04-08)
---------------------

Added
^^^^^^
- First public release of Langformers.
- Core NLP pipelines.
- Documentation and README.md.