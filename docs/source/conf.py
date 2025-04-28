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
    "sphinx_favicon"
]

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "private-members": True,
    "special-members": "__init__",
    "inherited-members": True,
    "show-inheritance": True,
}

html_title = "Langformers Documentation"

html_theme_options = {
    "collapse_navigation": False,
    "navigation_depth": 2,
    "logo_only": True,
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

html_baseurl = "https://langformers.com/"

favicons = [
   {"rel": "shortcut icon", "href": "favicon.ico"},
   {"rel": "icon", "type": "image/svg+xml", "href": "favicon.svg"},
   {"rel": "icon", "type": "image/png", "href": "favicon-96x96.png", "sizes": "96x96"},
   {"rel": "apple-touch-icon", "sizes": "180x180", "href": "apple-touch-icon.png"}
]
