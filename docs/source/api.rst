Tasks
--------

tasks
^^^^^^^
.. autoclass:: langformers.tasks
   :members:
   :inherited-members:
   :show-inheritance:

Classifiers
-----------------------

HuggingFaceClassifier
^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: langformers.HuggingFaceClassifier
   :members:
   :inherited-members:
   :show-inheritance:


.. autoclass:: langformers.classifiers.huggingface_classifier.TrainingConfig
   :exclude-members: __init__, per_device_train_batch_size, per_device_eval_batch_size, learning_rate, num_train_epochs, save_total_limit, logging_dir, eval_strategy, save_strategy, save_steps, logging_steps, metric_for_best_model, load_best_model_at_end, report_to, run_name, output_dir, max_length, test_size, val_size, early_stopping_patience, early_stopping_threshold, logging_strategy
   :noindex:


LoadClassifier
^^^^^^^^^^^^^^^^
.. autoclass:: langformers.classifiers.LoadClassifier
   :members:
   :inherited-members:
   :show-inheritance:


Embedders
------------

HuggingFaceEmbedder
^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: langformers.embedders.HuggingFaceEmbedder
   :members:
   :inherited-members:
   :show-inheritance:


Generators
------------

OllamaGenerator
^^^^^^^^^^^^^^^^^
.. autoclass:: langformers.generators.OllamaGenerator
   :members:
   :inherited-members:
   :show-inheritance:


HuggingFaceGenerator
^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: langformers.generators.HuggingFaceGenerator
   :members:
   :inherited-members:
   :show-inheritance:


StreamProcessor
^^^^^^^^^^^^^^^^^
.. autoclass:: langformers.generators.StreamProcessor
   :members:
   :inherited-members:
   :show-inheritance:


Labellers
------------

HuggingFaceDataLabeller
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: langformers.labellers.HuggingFaceDataLabeller
   :members:
   :inherited-members:
   :show-inheritance:


OllamaDataLabeller
^^^^^^^^^^^^^^^^^^^^
.. autoclass:: langformers.labellers.OllamaDataLabeller
   :members:
   :inherited-members:
   :show-inheritance:


Mimickers
------------

EmbeddingMimicker
^^^^^^^^^^^^^^^^^^^
.. autoclass:: langformers.mimickers.EmbeddingMimicker
   :members:
   :inherited-members:
   :show-inheritance:

.. autoclass:: langformers.mimickers.embedding_mimicker.StudentConfig
   :no-index:
   :exclude-members: __init__, hidden_size, intermediate_size, max_position_embeddings, num_attention_heads, num_hidden_layers
   :inherited-members:
   :show-inheritance:

.. autoclass:: langformers.mimickers.embedding_mimicker.TrainingConfig
   :no-index:
   :exclude-members: __init__, num_train_epochs, learning_rate, batch_size, dataset_path, logging_steps
   :inherited-members:
   :show-inheritance:

MLMs
------

MLMTokenizerDatasetCreator
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: langformers.mlms.MLMTokenizerDatasetCreator
   :members:
   :inherited-members:
   :show-inheritance:


.. autoclass:: langformers.mlms.mlm_tokenizer_dataset_creator.TokenizerConfig
   :no-index:
   :exclude-members: __init__, max_length, min_frequency, path_to_save_tokenizer, special_tokens, vocab_size
   :inherited-members:
   :show-inheritance:


HuggingFaceMLMCreator
^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: langformers.mlms.HuggingFaceMLMCreator
   :members:
   :inherited-members:
   :show-inheritance:


.. autoclass:: langformers.mlms.mlm_trainer.TrainingConfig
   :no-index:
   :exclude-members: __init__, per_device_train_batch_size, gradient_accumulation_steps, learning_rate, num_train_epochs, save_strategy, save_steps, logging_steps, save_total_limit, run_name, output_dir, logging_dir, report_to, n_gpus, mlm_probability, warmup_ratio
   :inherited-members:
   :show-inheritance:


Searchers
------------

FaissSearcher
^^^^^^^^^^^^^^^
.. autoclass:: langformers.searchers.FaissSearcher
   :exclude-members: create_faiss_index, init_db, load_or_create_index, close, get_text_by_id, save_index, train_index, get_by_id
   :inherited-members:
   :show-inheritance:

ChromaDBSearcher
^^^^^^^^^^^^^^^^^^
.. autoclass:: langformers.searchers.ChromaDBSearcher
   :exclude-members: close, initialize_collection
   :inherited-members:
   :show-inheritance:

PineconeSearcher
^^^^^^^^^^^^^^^^^^
.. autoclass:: langformers.searchers.PineconeSearcher
   :exclude-members: initialize_index
   :inherited-members:
   :show-inheritance: