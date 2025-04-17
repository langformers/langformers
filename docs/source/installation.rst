Installation
==============

Langformers requires **Python 3.10 or later**. Its installation is simple!

.. note::

   It's highly recommended to install Langformers in a **virtual environment** to avoid dependency conflicts. Follow the :ref:`platform-specific-instructions`.

To install Langformers globally:

.. code-block:: bash

   pip install -U langformers

This installs the latest version with :doc:`core dependencies <dependencies>`. If you need support for specific features (e.g., FAISS for vector search), you can install optional dependencies as well.

Optional Dependencies
------------------------

Langformers includes optional integrations you can install depending on your use case:

- For **FAISS** support: ``pip install -U langformers[faiss]``
- For **ChromaDB** support: ``pip install -U langformers[chromadb]``
- For **Pinecone** support: ``pip install -U langformers[pinecone]``

- To install **all optional features**: ``pip install -U langformers[all]``

.. important::

   If you use **zsh**, wrap everything after ``pip install`` in quotes to avoid shell interpretation. Example:

   .. code-block:: bash

      pip install "langformers[faiss]"

   This applies to `zsh` and any shell that might treat `[]` as a glob pattern.

.. _platform-specific-instructions:

Platform-specific Instructions
-------------------------------

.. tabs::

    .. tab:: Linux

        1. Verify that Python 3.10+ and pip are installed:

         .. code-block:: bash

            python3 --version
            pip3 --version

        2. If not installed (for Debian/Ubuntu-based distros):

         .. code-block:: bash

            sudo apt update
            sudo apt install python3.10 -y
            sudo apt install python3-pip -y

        3. Install `python3.10-venv` for virtual environment support:

         .. code-block:: bash

            sudo apt install python3.10-venv

        4. Create a virtual environment:

         .. code-block:: bash

            python3.10 -m venv env   # "env" is the environment name

        5. Activate the environment:

         .. code-block:: bash

            source env/bin/activate

        6. Install Langformers (and any extras):

         .. code-block:: bash

            pip install -U langformers


    .. tab:: macOS

        1. Verify Python 3.10+ and pip:

         .. code-block:: bash

            python3 --version
            pip3 --version

        2. If not installed, use Homebrew:

         .. code-block:: bash

            brew install python@3.10

        3. Create a virtual environment:

         .. code-block:: bash

            python3.10 -m venv env

        4. Activate the environment:

         .. code-block:: bash

            source env/bin/activate

        5. Install Langformers (and any extras):

         .. code-block:: bash

            pip install -U langformers


    .. tab:: Windows

        1. Verify Python 3.10+ and pip:

         .. code-block:: bash

            python --version
            pip --version

        If not installed, download from: https://www.python.org/downloads/

        2. Create a virtual environment:

         .. code-block:: bash

            python -m venv env

        3. Activate the environment:

         .. code-block:: bash

            env\Scripts\activate

        4. Install Langformers (and any extras):

         .. code-block:: bash

            pip install -U langformers


You're now ready to use Langformers! 🚀
