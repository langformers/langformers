Pretrain MLMs
================

Masked Language Models (MLMs) like BERT [Devlin2019]_, RoBERTa [Liu2019]_, and MPNet [Song2020]_ are auto-encoding
models that use only the encoder component of the Transformer [Vaswani2017]_ architecture. These models are ideal
for tasks requiring the entire input sequence to make decisions, such as text classification
and named-entity recognition.

Langformers adopts the RoBERTa architecture [Liu2019]_ as the default for both MLM pretraining and :doc:`model mimicking <mimick-a-model>` tasks.
This design choice enables a unified and streamlined training pipeline. RoBERTa is well-suited for both pretraining
and downstream fine-tuning or transfer tasks. Additionally, the architecture is highly customizable and easy to
understand, with configurable parameters such as the number of layers, hidden size, and attention heads.

.. tip::
    Pretraining an MLM is mostly beneficial when no domain-specific MLM exists. For example,
    BERTweet [Nguyen2020]_ was created for tweets, and CrisisTransformers [Lamsal2024]_ were developed for
    crisis-related tweets.

    Maybe you would want to continue pretraining an exsiting MLM like RoBERTa on your domain-specific dataset. Further
    pretraining produces great results [Lamsal2024]_. Check out Langformers' :doc:`Further Pretrain MLMs <further-pretrain-mlms>` task.

There are three steps to training an MLM from scratch. First, train a tokenizer on your dataset. Second, use
the trained tokenizer to tokenize the dataset. Finally, train the MLM on the tokenized dataset.

Let's get started!

Tokenizer and Tokenization
-----------------------------

Before training, you need to create a tokenizer (if you already don't have one) and tokenize your dataset. The tokenizer converts raw text into tokens that the model can process.

.. code-block:: python

   # Import langformers
   from langformers import tasks

   # Define configuration for the tokenizer
   tokenizer_config = {
        "vocab_size": 50_265,
        "min_frequency": 2,
        "max_length": 512,
        # ...
   }

   # Train the tokenizer and tokenize the dataset
   tokenizer = tasks.create_tokenizer(data_path="data.txt", tokenizer_config=tokenizer_config)
   tokenizer.train()

The trained tokenizer will be saved inside "tokenizer" and the tokenized dataset inside "tokenized_dataset" in the working directory.

.. tabs::

    .. tab:: create_tokenizer()

        .. autofunction:: langformers.tasks.create_tokenizer
           :no-index:

    .. tab:: tokenizer_config

        .. autoclass:: langformers.mlms.mlm_tokenizer_dataset_creator.TokenizerConfig
            :no-index:
            :exclude-members: __init__, max_length, min_frequency, path_to_save_tokenizer, special_tokens, vocab_size


Initialize an MLM and Train
-----------------------------------

With the tokenizer and dataset ready, initialize the MLM model and start training.

.. code-block:: python

   # Define model architecture
   model_config = {
       "vocab_size": 50_265,              # Size of the vocabulary (must match tokenizer's `vocab_size`)
       "max_position_embeddings": 514,    # !imp Maximum sequence length (tokenizer's `max_length` + 2)
       "num_attention_heads": 12,         # Number of attention heads
       "num_hidden_layers": 12,           # Number of hidden layers
       "hidden_size": 768,                # Size of the hidden layers
       "intermediate_size": 3072,         # Size of the intermediate layer in the Transformer
       # ...
   }

   # Define training configuration
   training_config = {
       "per_device_train_batch_size": 4,  # Batch size during training (per device)
       "num_train_epochs": 2,             # Number of training epochs
       "save_total_limit": 1,             # Maximum number of checkpoints to save
       "learning_rate": 2e-4,             # Learning rate for optimization
       # ...
   }

   # Initialize the training
   model = tasks.create_mlm(
       tokenizer="/path/to/tokenizer",
       tokenized_dataset="/path/to/tokenized_dataset",
       training_config=training_config,
       model_config=model_config
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
            from the specified path is resumed for pretraining. The latter is particularly useful for addressing
            issues  in the current checkpoint’s behavior\ [#]_ or :doc:`continuing the pretraining of an existing MLM <further-pretrain-mlms>`.

    .. tab:: model_config

        .. autoclass:: langformers.mlms.mlm_trainer.ModelConfig
            :no-index:
            :exclude-members: __init__, model_config


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
.. [Devlin2019] Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019). Bert: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 conference of the North American chapter of the association for computational linguistics: human language technologies, volume 1 (long and short papers) (pp. 4171-4186).
.. [Song2020] Song, K., Tan, X., Qin, T., Lu, J., & Liu, T. Y. (2020). Mpnet: Masked and permuted pre-training for language understanding. Advances in neural information processing systems, 33, 16857-16867.
.. [Vaswani2017] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. Advances in neural information processing systems, 30.
.. [Nguyen2020] Nguyen, D. Q., Vu, T., & Nguyen, A. T. (2020). BERTweet: A pre-trained language model for English Tweets. arXiv preprint arXiv:2005.10200.
.. [Lamsal2024] Lamsal, R., Read, M. R., & Karunasekera, S. (2024). CrisisTransformers: Pre-trained language models and sentence encoders for crisis-related social media texts. Knowledge-Based Systems, 296, 111916.

**Footnotes**

.. [#] When training MLMs, we typically simulate a larger batch size using gradient accumulation and multiple GPUs. As a result, a higher learning rate, such as 0.0004, is recommended. However, training loss may occasionally spike. In such cases, it may be necessary to halve the current learning rate, revert to the last stable checkpoint, and resume training.









