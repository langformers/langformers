Chunking
===========

.. raw:: html
   :file: ./_includes/official_links.html

Chunking breaks down large documents into smaller, manageable pieces (or "chunks"). This is especially important when dealing with documents that exceed the token limits of your embedding model. When designing RAG pipelines, it's crucial to consider the chunking strategy to ensure that the model can effectively process and retrieve relevant information.

There are many ways to chunk documents, and the best approach depends on the specific use case. Langformers offers following chunking strategies:

- Fixed-size chunking
- Semantic chunking
- Recursive chunking

.. admonition:: Tokenizer, tokens and chunks
    :class: warning
    
    Across all chunking strategies, the chunker uses the provided tokenizer to tokenize the document. The chunk size is defined in terms of tokens (not words). The number of tokens for a same document can vary depending on the tokenizer used.


Fixed-size chunking
---------------------

This approach divides a document into fixed-size chunks.

Here's a simple example of how to use the fixed-size chunker:

.. code-block:: python

    # Import Langformers
    from langformers import tasks

    # Create a chunker
    chunker = tasks.create_chunker(strategy="fixed_size", tokenizer="sentence-transformers/all-mpnet-base-v2")

    # Chunk a document
    chunks = chunker.chunk(document="This is a test document. It contains several sentences. We will chunk it into smaller pieces.",
                            chunk_size=8)

**Overlapping**: If ``overlap`` is provided to ``chunk()``, the chunker will create overlapping chunks.

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

Semantic chunking is a more advanced approach that uses semantic similarity to create chunks\ [#]_. Intially the document will be chunked into smaller pieces, and then the chunker will group them based on their semantic similarity.

Here's a simple example of how to use the semantic chunker:

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

Recursive chunking
---------------------
Recursive chunking divides a document hierarchically using specified separators.

Typically, a document is split by sections first, then paragraphs, and so on. Langformers follows this approach by initially splitting the text using ``\n\n`` (representing sections), then ``\n`` (for paragraphs), and finally down to the token level. At each stage, if a chunk exceeds the tokenizer’s maximum token limit, it is recursively split into smaller parts until each chunk fits within the allowed token size.

You can also declare your own separators to split the document.

Here's a simple example of how to use the recursive chunker:

.. code-block:: python

    # Import Langformers
    from langformers import tasks

    # Create a chunker
    chunker = tasks.create_chunker(strategy="recursive", tokenizer="sentence-transformers/all-mpnet-base-v2")

    # Chunk a document
    chunks = chunker.chunk(document="Cats are awesome.\n\nDogs are awesome.\nPython is amazing.",
                            separators=["\n\n", "\n"],
                            chunk_size=5)


.. tabs::

    .. tab:: create_chunker()

        .. autofunction:: langformers.tasks.create_chunker
           :no-index:

        **kwargs** for **recursive** strategy:

        .. autofunction:: langformers.chunkers.RecursiveChunker.__init__
           :no-index:

    .. tab:: chunk()

        .. autofunction:: langformers.chunkers.RecursiveChunker.chunk
           :no-index:


**Footnotes**

.. [#] Based on the concept presented by Greg Kamradt. The 5 Levels Of Text Splitting For Retrieval. https://www.youtube.com/watch?v=8OJC21T2SL4&t=1930s