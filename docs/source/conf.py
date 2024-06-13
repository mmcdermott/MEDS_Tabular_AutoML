import os
import sys

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "MEDS-Tab"
copyright = "2024, Nassim Oufattole, Matthew McDermott, Teya Bergamaschi, Aleksia Kolo, Hyewon Jeong"
author = "Nassim Oufattole, Matthew McDermott, Teya Bergamaschi, Aleksia Kolo, Hyewon Jeong"
release = "0.0.2"
version = "0.0.2"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# -- Path setup
from pathlib import Path

__location__ = Path(os.path.dirname(__file__))
__src__ = __location__ / "../.."

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
sys.path.insert(0, str(__src__))

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx_rtd_theme",
    "recommonmark",
    # "sphinx_immaterial"
]

source_suffix = {
    ".rst": "restructuredtext",
    ".txt": "markdown",
    ".md": "markdown",
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
}
intersphinx_disabled_domains = ["std"]

templates_path = ["_templates"]
exclude_patterns = []

autosummary_generate = True

pygments_style = "tango"


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = "sphinx_rtd_theme"
html_theme = "piccolo_theme"
# html_theme = "sphinx_immaterial"
html_static_path = ["_static"]


html_title = f"MEDS-Tab v{version} Documentation"
html_short_title = "MEDS-Tab Documentation"

# html_logo = "query-512.png"
# html_favicon = "query-16.ico"

# html_sidebars = {"**": ["logo-text.html", "globaltoc.html", "localtoc.html", "searchbox.html"]}

html_theme_options = {
    "dark_mode_code_blocks": False,
    # "nav_title": "MEDS-Tab",
    # "palette": {"primary": "green", "accent": "green"},
    # "repo_url": "https://github.com/mmcdermott/MEDS_Tabular_AutoML",
    # "repo_name": "MEDS_Tabular_AutoML",
    # # Visible levels of the global TOC; -1 means unlimited
    # "globaltoc_depth": 3,
    # If False, expand all TOC entries
    "globaltoc_collapse": True,
    # If True, show hidden TOC entries
    "globaltoc_includehidden": False,
}


html_show_copyright = True
htmlhelp_basename = "meds-tab-doc"


# -- Options for EPUB output
epub_show_urls = "footnote"
