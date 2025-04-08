# Langformers

Langformers is a flexible and user-friendly library that unifies NLP pipelines for both Large Language Models (LLMs) and Masked Language Models (MLMs) into one simple API.

Why Langformers? **Chat, build, train, label, and embed — faster than ever.**

Whether you're generating text, training classifiers, labelling data, embedding sentences, or building a semantic search index... the API stays consistent:

```python
from langformers import tasks

component = tasks.create_<something>(...)
component.<do_something>()
```

No need to juggle different frameworks — Langformers wraps Hugging Face Transformers, SentenceTransformers, Ollama, FAISS, ChromaDB, Pinecone, and more under one unified interface.

Use the same pattern everywhere:

```python
tasks.create_generator(...)  # Chatting with LLMs
tasks.create_labeller(...)   # Data labelling using LLMs
tasks.create_embedder(...)   # Embeding Sentences
tasks.create_classifier(...) # Training a Text Classifier
tasks.create_tokenizer()     # Training a Custom Tokenizer
tasks.create_mlm(...)        # Pretraining an MLM
tasks.create_searcher(...)   # Vector Database search
tasks.create_mimicker(...)   # Knowledge Distillation
```

## Supported Tasks

### Generative LLMs (e.g., Llama, Mistral, DeepSeek)
- Seamless Chat with LLMs
- LLM Inference via API
- Data Labelling with LLMs

### Masked Language Models (e.g., RoBERTa)
- Train Text Classifiers
- Pretrain MLMs from scratch
- Continue Pretraining on custom data

### Embeddings & Search (e.g., Sentence Transformers, FAISS, Pinecone)
- Embed Sentences
- Semantic Search
- Mimic a Pretrained Model (Knowledge Distillation)


## Installation

```bash
pip install -U langformers
```

## Quick Start
For more advanced use, refer to the documentation at https://langformers.com.

To get started quickly, below are some example use cases of Langformers.  


### Conversational AI 
Run this code as a python script (e.g., chat.py).
  
```python  
# Import langformers
from langformers import tasks

# Create a generator
generator = tasks.create_generator(provider="ollama", model_name="llama3.1:8b")

# Run the generator
generator.run(host="0.0.0.0", port=8000)  
```  

Open your browser at http://0.0.0.0:8000 (or the specific host and port you provided) to chat with the LLM.  
  
Instead of using the chat interface, if you want to perform LLM inference through a REST API, you can send a POST request to `host:port/api/generate` endpoint. This is great when you’re building your own application.  
  
The `host:port/api/generate` endpoint accepts the following:  
  
```json  
{  
 "system_prompt": "You are an Aussie AI assistant, reply in an Aussie way.",
 "memory_k": 10, 
 "temperature": 0.5, 
 "top_p": 1, 
 "max_length": 5000, 
 "prompt": "Hi"
 }  
``` 
  
### Data Labelling with LLMs  
Generative LLMs are highly effective for data labeling, extending beyond just conversation. Langformers offers the simplest way to define labels and conditions for labelling texts with LLMs.


```python  
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
```  
  
### Training a Text Classifier  
Training text classifiers with Langformers is quite straightforward.

First, we define the training configurations, prepare the dataset, and select the MLM we would like to fine-tune for the classification task. All these can be achieved in few lines of code, but fully customizable!

```python  
# Import langformers
from langformers import tasks

# Define training configuration
training_config = {
    "max_length": 80,
    "num_train_epochs": 1,
    "report_to": ['tensorboard'],
    "logging_steps": 20,
    "save_steps": 20,
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
```  
  
### Training a Custom Tokenizer
Before an MLM pretraining, you need to create a tokenizer (if you already don’t have one) and tokenize your dataset.

```python  
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
```  
  
### Pretraining an MLM 
With a tokenizer and tokenized dataset ready, pretraining an MLM is too easy with Langformers.


```python  
# Import langformers
from langformers import tasks

# Define model architecture
model_config = {
    "vocab_size": 50_265,            # Size of the vocabulary (must match tokenizer's `vocab_size`)
    "max_position_embeddings": 512,  # Maximum sequence length (must match tokenizer's `max_length`)
    "num_attention_heads": 12,       # Number of attention heads
    "num_hidden_layers": 12,         # Number of hidden layers
    "hidden_size": 768,              # Size of the hidden layers
    "intermediate_size": 3072,       # Size of the intermediate layer in the Transformer
    # ...
}

# Define training configuration
training_config = {
    "num_train_epochs": 2,           # Number of training epochs
    "save_total_limit": 1,           # Maximum number of checkpoints to save
    "learning_rate": 2e-4,           # Learning rate for optimization
    # ...
}

# Initialize the training
model = tasks.create_mlm(
    tokenizer="tokenizer",
    tokenized_dataset="tokenized_dataset",
    training_config=training_config,
    model_config=model_config
)

# For continuing pretraining of a existing MLM such as RoBERTa
# provide `checkpoint_path` to tasks.create_mlm() instead of `model_config`.

# Start the training
model.train()  
```  
  
### Embed Sentences 
Using state-of-the-art embedding models for vectorizing your sentences takes just two steps with Langformers.

```python  
# Import langformers
from langformers import tasks

# Create an embedder
embedder = tasks.create_embedder(provider="huggingface", model_name="sentence-transformers/all-MiniLM-L6-v2")

# Get your sentence embeddings
embeddings = embedder.embed(["I am hungry.", "I want to eat something."])
```  
  
### Semantic Search  
Langformers can help you quickly set up a semantic search engine for vectorized text retrieval. All you need to do is specify an embedding model, the type of database (FAISS, ChromaDB, or Pinecone), and an index type (if required).


```python  
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
```  
  
### Knowledge Distillation (Mimicking a pretrained model)  
Langformers can train a custom model to replicate the embedding space of a pretrained teacher model.

```python  
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

# Define the training configurations
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
```  



## Documentation
For full documentation, API reference, and advanced usage, visit: https://langformers.com



## License  
Langformers is released under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).



## Contributing  
We welcome contributions! Please see our contribution guidelines for details.



Built with ❤️ for the future of language AI.