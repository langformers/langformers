Installation
==============

Installing Langformers is simple and can be done in just a few steps. Follow the instructions for your operating system below.

To install Langformers globally, run:

.. code-block:: bash

   pip install langformers

However, it is strongly recommended to use a virtual environment for better package management.

.. tabs::

    .. tab:: Linux

        1. Ensure `Python` and `pip` are installed on your machine. To verify:

         .. code-block:: bash

            python3 --version
            pip3 --version

        2. If they are not installed, use `apt` to install them (for Debian-based distros):

         .. code-block:: bash

            sudo apt update
            sudo apt install python3 -y
            sudo apt install python3-pip -y

        3. Install `python3-venv` to create a virtual environment:

         .. code-block:: bash

            sudo apt install python3-venv

        4. Create a new Python virtual environment:

         .. code-block:: bash

            python3 -m venv env   # `env` is the environment name

        5. Activate the virtual environment:

         .. code-block:: bash

            source env/bin/activate

        6. Install Langformers inside the environment:

         .. code-block:: bash

            pip install langformers


    .. tab:: Windows

        1. Ensure `Python` and `pip` are installed on your machine. To verify:

         .. code-block:: bash

            python --version
            pip --version

         If Python and pip are not installed, download Python from the official website:

         https://www.python.org/downloads/

        2. Create a new Python virtual environment:

         .. code-block:: bash

            python -m venv env   # `env` is the environment name

        3. Activate the virtual environment:

         .. code-block:: bash

            env\Scripts\activate

        4. Install Langformers inside the environment:

         .. code-block:: bash

            pip install langformers


    .. tab:: macOS

        1. Ensure `Python` and `pip` are installed on your machine. To verify:

         .. code-block:: bash

            python3 --version
            pip3 --version

        2. If `Python` is not installed, use Homebrew to install it:

         .. code-block:: bash

            brew install python3

        3. Create a new Python virtual environment:

         .. code-block:: bash

            python3 -m venv env   # `env` is the environment name

        4. Activate the virtual environment:

         .. code-block:: bash

            source env/bin/activate

        5. Install Langformers inside the environment:

         .. code-block:: bash

            pip install langformers


Now you're ready to use Langformers! 🎉
