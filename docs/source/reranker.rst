Rerank Sentences
==================

Langformers also supports reranking. Reranking reorders a list of documents (or sentences/texts) based on their relevance to a given query.

Vector search may not always yield the most relevant results. Reranking can help improve the quality of the retrieved documents by reordering them based on their relevance to the query.

.. admonition:: Vector search and Reranking
        :class: tip

        For instance, suppose a user searches for "Where is Mount Everest?", and our database contains the following documents:

        1. "Mount Everest is the highest mountain in the world."
        2. "Mount Everest is in Nepal."
        3. "Where is Mount Everest?"

        A basic vector search might rank the third document highest because it closely matches the wording of the query — that's what semantic similarity (like cosine similarity between embeddings) picks up on. But here's the problem: that document just repeats the question instead of answering it.

        This is where reranking comes in. It reorders the documents based on their relevance to the query so that the most appropriate document (i.e., the second one) appears at the top of the list.

Here's a sample code snippet to get you started:


.. code-block:: python

    # Import langformers
    from langformers import tasks

    # Create a reranker
    reranker = tasks.create_reranker(model_type="cross_encoder", model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")

    # Define `query` and `documents`
    query = "Where is the Mount Everest?"

    documents = [
        "Mount Everest is the highest mountain in the world.",
        "Mount Everest is in Nepal.",
        "Where is the Mount Everest?"
    ]
    
    # Get your reranked documents
    reranked_docs = reranker.rank(query=query, documents=documents)
    print(reranked_docs)


.. tabs::

    .. tab:: create_reranker()

        .. autofunction:: langformers.tasks.create_reranker
           :no-index:

        **kwargs** for **Cross Encoder** model type:

        .. autoclass:: langformers.rerankers.CrossEncoder
           :exclude-members: rank
           :inherited-members:
           :show-inheritance:
           :no-index:

    .. tab:: rank()

        .. autofunction:: langformers.rerankers.CrossEncoder.rank
           :no-index: