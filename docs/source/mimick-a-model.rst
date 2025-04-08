Mimick a Pretrained Model
============================

Langformers can train a custom model to replicate the embedding space
of a pretrained teacher model.

This task is inspired from the works [Hinton2015]_, [Reimers2020]_, [Lamsal2025]_, where a student model
learns to match the output embeddings of the teacher model by minimizing the mean squared error (MSE) loss between
the teacher's and student's embeddings.

.. admonition:: When to mimick?
    :class: warning

    Mimicking the vector space of a pretrained model is especially useful when creating smaller models. For example,
    if you have a high-performing model like "roberta-base" or "sentence-transformers/all-mpnet-base-v2," you can create a
    smaller model that attempts to replicate the outputs of these models. The best part is that you can customize the
    architecture of your smaller model to suit your specific needs.

.. note::

    💡 The idea is to pass a large number of sentences through both the teacher and the student models, then adjust
    the student's weights to align with those of the teacher. This approach allows the student to mimic the teacher's
    vector space.

    Training requires a corpus of sentences. Langformers provides two ready-to-use datasets:

    - For general purpose models\ [#]_: https://huggingface.co/datasets/Langformers/allnli-mimic-embedding
    - For social media models\ [#]_: https://huggingface.co/datasets/Langformers/sentiment140-mimic-embedding

To mimick a pretrained model with your custom transformer architecture, create a mimicker with ``create_mimicker()`` and ``train()`` it.

Here's a sample code for you to get started:

.. code-block:: python

    # Load a text corpus
    # In this example we use all the sentences from `allnli` dataset.
    from datasets import load_dataset
    data = load_dataset("langformers/allnli-mimic-embedding")

    # Import langformers
    from langformers import tasks

    # Define the architecture of your student model
    student_config = {
        "max_position_embeddings": 130,
        "num_attention_heads":8,
        "num_hidden_layers": 8,
        "hidden_size": 128,
        "intermediate_size": 256,
        # ...
    }

    # Define training configuration
    training_config = {
        "num_train_epochs": 10,
        "learning_rate": 5e-5,
        "batch_size": 128,                          # use large batch
        "dataset_path": data['train']['sentence'],  # `list` of sentences or `path` to a text corpus
        "logging_steps": 100,
        # ...
    }

    # Create a mimicker
    mimicker = tasks.create_mimicker(teacher_model="roberta-base", student_config=student_config, training_config=training_config)

    # Start training
    mimicker.train()

That's all! Every logging steps, if the loss improves, the checkpoint is saved. So, you'll always have the best mimicker.

.. note::

    The mimicker will use the same vocabulary as the teacher model, meaning it will employ the same tokenizer.


.. tabs::

    .. tab:: create_mimicker()

        .. autofunction:: langformers.tasks.create_mimicker
           :no-index:

    .. tab:: student_config

        .. autoclass:: langformers.mimickers.embedding_mimicker.StudentConfig
           :no-index:
           :exclude-members: __init__, hidden_size, intermediate_size, max_position_embeddings, num_attention_heads, num_hidden_layers
           :inherited-members:
           :show-inheritance:


    .. tab:: training_config

        .. autoclass:: langformers.mimickers.embedding_mimicker.TrainingConfig
           :no-index:
           :exclude-members: __init__, num_train_epochs, learning_rate, batch_size, dataset_path, logging_steps
           :inherited-members:
           :show-inheritance:


**References**

.. [Hinton2015] Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the knowledge in a neural network. arXiv preprint arXiv:1503.02531.
.. [Reimers2020] Reimers, N., & Gurevych, I. (2020). Making monolingual sentence embeddings multilingual using knowledge distillation. arXiv preprint arXiv:2004.09813.
.. [Lamsal2025] Lamsal, R., Read, M. R., Karunasekera, S., & Imran, M. (2025). "Actionable Help" in Crises: A Novel Dataset and Resource-Efficient Models for Identifying Request and Offer Social Media Posts. arXiv preprint arXiv:2502.16839.

**Footnotes**

.. [#] Pre-trained models for processing texts from general domains (e.g., `BERT <https://huggingface.co/google-bert/bert-base-uncased>`_, `RoBERTa <https://huggingface.co/FacebookAI/roberta-base>`_, `MPNet <https://huggingface.co/microsoft/mpnet-base>`_).
.. [#] Pre-trained models for processing social media texts (e.g., `BERTweet <https://huggingface.co/vinai/bertweet-base>`_, `CrisisTransformers <https://huggingface.co/crisistransformers>`_).


