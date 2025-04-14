Chunking
===========

Chunking breaks down large documents into smaller, manageable pieces (or "chunks"). This is especially important when dealing with documents that exceed the token limits of your embedding model. When designing RAG pipelines, it's crucial to consider the chunking strategy to ensure that the model can effectively process and retrieve relevant information.

There are many ways to chunk documents, and the best approach depends on the specific use case. Langformers offers following chunking strategies:

.. admonition:: Tokenizer, tokens and chunks
    :class: warning
    
    Across all chunking strategies, the chunker uses the provided tokenizer to tokenize the document. The chunk size is defined in terms of tokens (not words). The number of tokens for a same document can vary depending on the tokenizer used.


Fixed-size chunking
---------------------

This approach divides a document into fixed-size chunks.

.. code-block:: python

    # Import Langformers
    from langformers import tasks

    # Create a chunker
    chunker = tasks.create_chunker(strategy="fixed_size", tokenizer="sentence-transformers/all-mpnet-base-v2")

    # Chunk a document
    chunks = chunker.chunk(document="This is a test document. It contains several sentences. We will chunk it into smaller pieces.",
                            chunk_size=8)

**Overlapping**: If ``overlap`` is provided to ``chunk()``, the chunker will create overlapping chunks.

Also, instead of getting chunks as a list, you can save them to a file by providing the ``save_as`` parameter.

.. tabs::

    .. tab:: create_chunker()

        .. autofunction:: langformers.tasks.create_chunker
           :no-index:

        **kwargs** for **fixed_size** strategy:
        
        .. autofunction:: langformers.chunkers.FixedSizeChunker.__init__
           :no-index:

    .. tab:: chunk()

        .. autofunction:: langformers.chunkers.FixedSizeChunker.chunk
           :no-index:


Semantic Chunking
--------------------

Semantic chunking is a more advanced approach that uses semantic similarity to create chunks. Intially the document will be chunked into smaller pieces, and then the chunker will group them based on their semantic similarity.

.. code-block:: python

    # Import Langformers
    from langformers import tasks

    # Create a chunker
    chunker = tasks.create_chunker(strategy="semantic", model_name="sentence-transformers/all-mpnet-base-v2")

    # Chunk a document
    chunks = chunker.chunk(document="Cats are awesome. Dogs are awesome. Python is amazing.", 
                            initial_chunk_size=4,
                            max_chunk_size=10,
                            similarity_threshold=0.3)


.. tabs::

    .. tab:: create_chunker()

        .. autofunction:: langformers.tasks.create_chunker
           :no-index:

        **kwargs** for **semantic** strategy:

        .. autofunction:: langformers.chunkers.SemanticChunker.__init__
           :no-index:

    .. tab:: chunk()

        .. autofunction:: langformers.chunkers.SemanticChunker.chunk
           :no-index: