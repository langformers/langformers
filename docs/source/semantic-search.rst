Semantic Search
=================

Langformers can help you quickly set up a semantic search engine for vectorized text retrieval. All you need to do
is specify an embedding model, the type of database (FAISS, ChromaDB, or Pinecone), and an index type (if required).

Here’s a sample code snippet to get you started:

.. code-block:: python

    # Import langformers
    from langformers import tasks

    # Initialize a searcher
    searcher = tasks.create_searcher(embedder="sentence-transformers/all-MiniLM-L12-v2", database="faiss", index_type="HNSW")

    '''
    For other vector databases:

    ChromaDB
    searcher = llms.create_searcher(embedder="sentence-transformers/all-MiniLM-L12-v2", database="chromadb")

    Pinecone
    searcher = llms.create_searcher(embedder="sentence-transformers/all-MiniLM-L12-v2", database="pinecone", api_key="your-api-key-here")
    '''

    # Sentences to add in the vector database
    sentences = [
        "He is learning Python programming.",
        "The coffee shop opens at 8 AM.",
        "She bought a new laptop yesterday.",
        "He loves to play basketball with friends.",
        "Artificial Intelligence is evolving rapidly.",
        "He studies CS at the University of Melbourne."
    ]

    # Metadata for the respective sentences
    metadata = [
        {"action": "learning", "category": "education"},
        {"action": "opens", "category": "business"},
        {"action": "bought", "category": "shopping"},
        {"action": "loves", "category": "sports"},
        {"action": "evolving", "category": "technology"},
        {"action": "studies", "category": "education"}
    ]

    # Add the sentences
    searcher.add(texts=sentences, metadata=metadata)

    # Define a search query
    query_sentence = "computer science"

    # Query the vector database
    results = searcher.query(query=query_sentence, items=2, include_metadata=True)
    print(results)

Loading an Existing Database
----------------------------
Once a searcher is initialized, the specified index/database is persisted on disk (if FAISS and ChromaDB) and on cloud (if Pinecone).
To load an existing database, initialize a searcher with the following parameters:

- For FAISS: ``index_path`` and ``db_path``.
- For ChromaDB: ``db_path`` and ``collection_name``.
- For Pinecone: ``index_name``

.. tabs::

    .. tab:: create_searcher()

        .. autofunction:: langformers.tasks.create_searcher
           :no-index:

        Database-specific **kwargs**

        **FAISS**

        .. autoclass:: langformers.searchers.FaissSearcher
           :exclude-members: create_faiss_index, init_db, load_or_create_index, close, get_text_by_id, save_index, train_index, add, count, query, get_by_id
           :inherited-members:
           :show-inheritance:
           :no-index:

        **ChromaDB**

        .. autoclass:: langformers.searchers.ChromaDBSearcher
           :exclude-members: close, add, count, query, initialize_collection
           :inherited-members:
           :show-inheritance:
           :no-index:

        **Pinecone**

        .. autoclass:: langformers.searchers.PineconeSearcher
           :exclude-members: add, count, query, initialize_index
           :inherited-members:
           :show-inheritance:
           :no-index:

    .. tab:: add()

        ``add()`` takes the following parameters:

        - ``texts`` (list[str], required): List of text entries to be indexed.
        - ``metadata`` (Optional[List[Dict[str, Any]]], default=None): Metadata associated with each text.

    .. tab:: query()

        ``query()`` takes the following parameters:

        - ``query`` (str, required): The input text query.
        - ``items`` (int, default=1): Number of nearest neighbors to retrieve.
        - ``include_metadata`` (bool, default=True): Whether to include the metadata in the results.

    .. tab:: count()

        ``count()`` does not take any parameters. Simply run ``<searcher object>.count()``.
