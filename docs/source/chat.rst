Seamless Chat with LLMs
==========================

Chatting with generative Large Language Models (LLMs) is easy with Langformers.

.. note::
    Chatting is supported for `Ollama` and `Hugging Face`\ [#]_ models. In the case of `Ollama`, please ensure that it is installed on your
    OS and that the desired model is pulled before starting a conversation.

    - Install Ollama: https://ollama.com/download
    - Pull model: For e.g., ``ollama pull llama3.1:8b`` on your terminal

.. admonition:: Running the code
    :class: warning

    The code below must be run as a Python script or executed in a terminal using ``python3``. It will not work inside notebooks.

Here’s a sample code snippet to get you started:

.. code-block:: python

    # Import langformers
    from langformers import tasks

    # Create a generator
    generator = tasks.create_generator(provider="ollama", model_name="llama3.1:8b")

    # Run the generator
    generator.run(host="0.0.0.0", port=8000)

Open your browser at ``http://0.0.0.0:8000`` (or the specific ``host`` and ``port`` you provided) to chat with the LLM.

.. tabs::

    .. tab:: create_generator()

        .. autofunction:: langformers.tasks.create_generator
           :no-index:

    .. tab:: run()

        ``run()`` takes the following parameters:

        - ``host`` (str, default="0.0.0.0"): The IP address to bind the server to.
        - ``port`` (int), default=8000: The port number to listen on.

    .. tab:: chat interface

        The chat interface has the following parameters.

        - ``system_prompt`` (str, default=<Langformers.commons.prompts default_chat_prompt_system>): The system-level instruction for the LLM.
        - ``memory_k`` (int, default=10): The number of previous messages to retain in memory.
        - ``temperature`` (float, default=0.5): Controls randomness of responses (higher = more random).
        - ``top_p`` (float, default=1): Nucleus sampling parameter (lower = more focused).
        - ``max_length`` (int, default=5000): Maximum number of tokens to generate.
        - ``authorization_token`` (string, default=None): Authorization token if the endpoint requires authentication.
        - ``prompt`` (str, required): User query.


Authentication
-----------------
Opening the chat user interface does not require authorization. Since the interface interacts with an LLM via ``/api/generate`` endpoint, in some cases we might want to protect the endpoint from unauthorized access. Securing the endpoint is straightforward. You can pass a dependency to ``dependency`` when creating the generator.

.. code-block:: python

    async def auth_dependency():
        """Authorization dependency for request validation.

        - Implement your own logic here (e.g., API key check, authentication).
        - If the function returns a value, access is granted.
        - Raising an HTTPException will block access.
        - Modify this logic as needed.
        """

    generator = tasks.create_generator(provider="ollama", model_name="llama3.1:8b", dependency=auth_dependency)


**Example: Using API Key Authentication**

You can implement a simple authentication dependency like this:

.. code-block::

    # Imports
    from langformers import tasks
    from fastapi import Request, HTTPException

    # Define a set of valid API keys
    API_KEYS = {"12345", "67890"}

    async def auth_dependency(request: Request):
        """
        Extracts the Bearer token and verifies it against a list of valid API keys.
        """
        auth_header = request.headers.get("Authorization")

        if not auth_header or not auth_header.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Invalid authorization header format.")

        token = auth_header.split("Bearer ")[1]
        if token not in API_KEYS:
            raise HTTPException(status_code=401, detail="Unauthorized.")

        return True  # Allow access

    # Create a generator with authentication
    generator = tasks.create_generator(provider="ollama", model_name="llama3.1:8b", dependency=auth_dependency)

    # Run the generator
    generator.run(host="0.0.0.0", port=8000)

Now, a valid authorization token (one of the API keys in this case) should be entered into the "Authorization Token" text box (located in the left sidebar) of the chatting interface to interact with the LLM.

.. warning::
    Langformers uses the ``Authorization: Bearer <token>`` header for the chat interface. For :doc:`LLM inference <llm-inference>`, you can implement your own header format and authentication logic.

    For industry-standard authentication in FastAPI, you can use OAuth2 with JWT (JSON Web Token), which is widely adopted for securing APIs.


**Footnotes**

.. [#] Hugging Face support is limited to chat-tuned models (instruct) that include a ``chat_template`` in their ``tokenizer_config.json`` and are compatible with the `transformers` library and your system's hardware.