Data Labeling using LLMs
==========================

Generative LLMs are highly effective for data labeling, extending beyond just conversation.
Langformers offers the simplest way to define `labels` and `conditions` for labelling texts with LLMs.

To label texts, first load an LLM as a data labeller with ``create_labeller()``, then apply ``label()``.

Here's a sample code for you to get started:

.. code-block:: python

    # Import langformers
    from langformers import tasks

    # Load an LLM as a data labeller
    labeller = tasks.create_labeller(provider="huggingface", model_name="meta-llama/Meta-Llama-3-8B-Instruct", multi_label=False)

    # Provide labels and conditions
    conditions = {
        "Positive": "The text expresses a positive sentiment.",
        "Negative": "The text expresses a negative sentiment.",
        "Neutral": "The text does not express any emotions."
    }

    # Label a text
    text = "No doubt, The Shawshank Redemption is a cinematic masterpiece."
    labeller.label(text, conditions)


.. tabs::

    .. tab:: create_labeller()

        .. autofunction:: langformers.tasks.create_labeller
           :no-index:

    .. tab:: label()

        ``label()`` takes the following parameters:

        - ``text`` (str, required): The text to be labelled.
        - ``conditions`` (Dict[str, str], required): A dictionary mapping labels to their descriptions.



