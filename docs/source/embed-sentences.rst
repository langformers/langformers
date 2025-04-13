Embed Sentences
=================

Using state-of-the-art embedding models for vectorizing your sentences takes just two steps with Langformers.
First, create an embedder with ``create_embedder()``, and then call ``embed()`` on it.

Here's a sample code snippet to get you started:

.. code-block:: python

    # Import langformers
    from langformers import tasks

    # Create an embedder
    embedder = tasks.create_embedder(provider="huggingface", model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Get your sentence embeddings
    embeddings = embedder.embed(["I am hungry.", "I want to eat something."])

.. tip::

    Which models to use? Refer to this Hugging Face page for the list of supported embedding models: https://huggingface.co/models?library=sentence-transformers.

.. tabs::

    .. tab:: create_embedder()

        .. autofunction:: langformers.tasks.create_embedder
           :no-index:

    .. tab:: embed()

        .. autofunction:: langformers.embedders.HuggingFaceEmbedder.embed
           :no-index:

Textual Similarity
---------------------

To compute textual similarity between two texts, use ``similarity()``.

.. code-block:: python

    # Get cosine similarity
    embedder.similarity(["I am hungry.", "I am starving."])

.. tabs::

    .. tab:: similarity()

        .. autofunction:: langformers.embedders.HuggingFaceEmbedder.similarity
           :no-index:



