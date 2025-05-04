Further Pretrain MLMs
=======================

.. raw:: html
   :file: ./_includes/official_links.html

Continuing the pretraining of a Masked Language Model (MLM) follows a similar approach to how we :doc:`pretrain
MLMs <pretrain-mlms>`. However, unlike standard pretraining, we don’t initialize the model with random weights.
Instead, there are two possible scenarios:

- You have a saved checkpoint of an MLM created by Langformers and want to continue training.
- You want to take an existing MLM and further pretrain it on your custom dataset. Further pretraining an existing MLM produces great results [Lamsal2024]_.

Model pretrained using Langformers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
If you :doc:`trained an MLM with Langformers <pretrain-mlms>`, you should already have a tokenizer and a tokenized dataset ready; simply provide ``checkpoint_path`` instead of ``model_config`` to ``create_mlm()``.

.. code-block:: python

   # Import Langformers
   from langformers import tasks

   # Define training configuration
   training_config = {
       "num_train_epochs": 2,             # Number of training epochs
       "save_total_limit": 1,             # Maximum number of checkpoints to save
       "learning_rate": 2e-4,             # Learning rate for optimization
       "per_device_train_batch_size": 4,  # Batch size during training (per device)
       # ...
   }

   # Initialize the training
   model = tasks.create_mlm(
       tokenizer="/path/to/tokenizer",
       tokenized_dataset="/path/to/tokenized_dataset",
       training_config=training_config,
       checkpoint_path = "path/to/checkpoint"
   )

   # Start the training
   model.train()


Existing Model from HuggingFace
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
However, if you want to train an existing MLM model such as RoBERTa, BERTweet, and CrisisTransformers (from Hugging Face), you'll have a trained tokenizer compatible with that model but you'll need to create a tokenized dataset on which you'll be further pretraining.

.. warning::

    Langformers supports further pretraining of only the models trained with RoBERTa pretraining procedure. Such
    as RoBERTa [Liu2019]_, BERTweet [Nguyen2020]_, or CrisisTransformers [Lamsal2024]_.

Here's how you can create a tokenized dataset:

.. code-block:: python

   # Import Langformers
   from langformers import tasks

   # Tokenize the dataset with existing tokenizer.
   # This example uses "roberta-base" tokenizer from Hugging Face.
   dataset = tasks.create_tokenizer(data_path="data.txt", tokenizer="roberta-base")
   dataset.train()

This saves the tokenized dataset inside "tokenized_dataset" in the working directory.

Next, we start the training.


.. code-block:: python

   # Import Langformers
   from langformers import tasks

   # Define training configuration
   training_config = {
       "per_device_train_batch_size": 4,  # Batch size during training (per device)
       "num_train_epochs": 2,             # Number of training epochs
       "save_total_limit": 1,             # Maximum number of checkpoints to save
       "learning_rate": 2e-4,             # Learning rate for optimization
       "per_device_train_batch_size": 4,  # Batch size
       # ...
   }

   # Initialize the training
   # this example further pretrains "roberta-base"
   model = tasks.create_mlm(
       tokenizer="roberta-base",
       tokenized_dataset="/path/to/tokenized_dataset",
       training_config=training_config,
       checkpoint_path = "roberta-base"
   )

   # Start the training
   model.train()

.. tabs::

    .. tab:: create_mlm()

        .. autofunction:: langformers.tasks.create_mlm
           :no-index:

        .. warning::

            At least one of ``model_config`` or ``checkpoint_path`` must be provided. If ``model_config`` is specified,
            a new model is initialized using the given configurations. If ``checkpoint_path`` is provided, the model
            from the specified path is resumed for pretraining.

    .. tab:: training_config

        .. autoclass:: langformers.mlms.mlm_trainer.TrainingConfig
           :no-index:
           :exclude-members: __init__, per_device_train_batch_size, gradient_accumulation_steps, learning_rate, num_train_epochs, save_strategy, save_steps, logging_steps, save_total_limit, run_name, output_dir, logging_dir, report_to, n_gpus, mlm_probability, warmup_ratio
           :inherited-members:
           :show-inheritance:

        .. admonition:: Training loss is the main metric
            :class: warning

            Langformers does not evaluate checkpoints from MLM pretraining on a separate evaluation split, as it is generally unnecessary. In MLM pretraining, training loss is the primary metric since the goal is to learn rich representations rather than minimize validation loss. Real performance is ultimately determined by fine-tuning on downstream tasks.



**References**

.. [Liu2019] Liu, Y., Ott, M., Goyal, N., Du, J., Joshi, M., Chen, D., ... & Stoyanov, V. (2019). Roberta: A robustly optimized bert pretraining approach. arXiv preprint arXiv:1907.11692.
.. [Nguyen2020] Nguyen, D. Q., Vu, T., & Nguyen, A. T. (2020). BERTweet: A pre-trained language model for English Tweets. arXiv preprint arXiv:2005.10200.
.. [Lamsal2024] Lamsal, R., Read, M. R., & Karunasekera, S. (2024). CrisisTransformers: Pre-trained language models and sentence encoders for crisis-related social media texts. Knowledge-Based Systems, 296, 111916.
