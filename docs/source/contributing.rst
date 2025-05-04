Contributing
===============

.. raw:: html
   :file: ./_includes/official_links.html

Thank you for considering a contribution to **Langformers**!
Whether it's a bug fix, a feature, or documentation improvement, your help is appreciated. 💙

This guide will walk you through the process of contributing to the project.

.. contents:: Table of Contents
   :depth: 2
   :local:

Setting Up the Project
-------------------------

1. Fork the repository on GitHub.
2. Clone your fork locally:

   .. code-block:: bash

      git clone https://github.com/your-username/langformers.git
      cd langformers

3. (Optional) Create and activate a virtual environment:

   .. code-block:: bash

      python -m venv env
      source env/bin/activate  # On Windows: env\Scripts\activate

4. Install dependencies:

   .. code-block:: bash

      pip install -r requirements.txt

5. Install the project in editable mode:

   .. code-block:: bash

      pip install -e .

Ways to Contribute
---------------------

- **Report Bugs:** Open an issue in our GitHub repository.
- **Suggest Features:** Submit an issue with the `enhancement` label.
- **Submit Code:** Follow the pull request instructions below.
- **Improve Docs:** Documentation fixes are always welcome!

Pull Request Guidelines
--------------------------

- Create a new branch:

  .. code-block:: bash

     git checkout -b feature/my-feature

- Follow `PEP 8 <https://peps.python.org/pep-0008/>`_ style guide.
- Ensure your code is well tested.
- Use clear and descriptive commit messages.
- Rebase your branch onto the latest `main` before submitting a PR.

Documentation
----------------

We use **Sphinx** for building documentation. To build the docs locally:

.. code-block:: bash

   cd docs
   pip install -r requirements.txt
   make clean && make html

Then open ``build/html/index.html`` in your browser.

License
----------

By contributing to this project, you agree that your contributions will be licensed under the project's existing license, i.e., `Apache License 2.0`_.

----

Thank you for being awesome!

.. _Apache License 2.0: https://www.apache.org/licenses/LICENSE-2.0