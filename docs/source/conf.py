import os
import sys
import subprocess
sys.path.insert(0, os.path.abspath("../../"))


project = "langformers"
copyright = "2025. Built with ❤️ for the future of language AI"
author = "Rabindra Lamsal"

try:
    version = subprocess.check_output(["git", "describe", "--tags"]).decode().strip()
except Exception:
    version = "0.0.0"

release = version

extensions = [
    "sphinx_tabs.tabs",
    "sphinx.ext.todo",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
]

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "private-members": True,
    "special-members": "__init__",
    "inherited-members": True,
    "show-inheritance": True,
}

html_theme_options = {
    "collapse_navigation": False,
    "navigation_depth": 2,
    "logo_only": True
}

html_context = {
    "display_github": True,
    "github_user": "langformers",
    "github_repo": "langformers",
    "github_version": "main", 
    "conf_py_path": "/docs/source/",
}

templates_path = ['_templates']
exclude_patterns = []

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_logo = "_static/logo.png"
html_css_files = ["custom.css"]
html_favicon = "_static/favicon.svg"
