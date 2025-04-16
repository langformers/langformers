Train Text Classifiers
=========================

Training text classifiers with Langformers is quite straightforward.

First, we define the training configurations, prepare the dataset, and select the MLM we would like to fine-tune for the classification task. All these can be achieved in few lines of code, but fully customizable!

Here's a sample code to getting started.

.. code-block:: python

    # Import langformers
    from langformers import tasks

    # Define training configuration
    training_config = {
        "max_length": 80,
        "num_train_epochs": 1,
        "report_to": ['tensorboard'],
        "logging_steps": 20,
        "save_steps": 20,
        "early_stopping_patience": 5,
        # ...
    }

    # Initialize the model
    model = tasks.create_classifier(
        model_name="roberta-base",          # model from Hugging Face or a local path
        csv_path="/path/to/dataset.csv",    # csv dataset
        text_column="text",                 # text column name
        label_column="label",               # label/class column name
        training_config=training_config
    )

    # Start fine-tuning
    model.train()

This fine-tunes the selected MLM (e.g., "roberta-base") automatically based on the number of classes identified in the training dataset.

At the end of training, the best model is saved along with its configurations.

.. admonition:: Labels/classes datatype
    :class: warning

    ``train()`` assumes that the labels/classes in your training dataset are formatted as strings (e.g., "positive", "neutral", "negative") rather than numeric values (e.g., "0", "1", "2").
    Using human-readable labels (instead of encoded numbers) makes the classifier more intuitive to use during inference, reducing potential confusion.

.. tabs::

    .. tab:: create_classifier()

        .. autofunction:: langformers.tasks.create_classifier
            :no-index:

    .. tab:: training_config

        .. autoclass:: langformers.classifiers.huggingface_classifier.TrainingConfig
           :exclude-members: __init__, per_device_train_batch_size, per_device_eval_batch_size, learning_rate, num_train_epochs, save_total_limit, logging_dir, eval_strategy, save_strategy, save_steps, logging_steps, metric_for_best_model, load_best_model_at_end, report_to, run_name, output_dir, max_length, test_size, val_size, early_stopping_patience, early_stopping_threshold, logging_strategy
           :noindex:


Using the Classifier
----------------------

You can load the trained classifier with ``load_classifier()``.

.. code-block:: python

    # Import langformers
    from langformers import tasks

    # Load the trained classifier
    classifier = tasks.load_classifier("/path/to/classifier")

    # Classify texts
    classifier.classify(["I dont like this movie. Worst ever.", "I loved this movie."])

.. tabs::

    .. tab:: load_classifier()

        .. autofunction:: langformers.tasks.load_classifier
            :no-index:

    .. tab:: classify()

        .. autofunction:: langformers.classifiers.load_classifier.LoadClassifier.classify
            :no-index:
