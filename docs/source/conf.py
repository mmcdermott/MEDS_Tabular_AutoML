import os
import shutil
import sys
from pathlib import Path

import MEDS_tabular_automl

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "MEDS-Tab"
copyright = "2024, Nassim Oufattole, Matthew McDermott, Teya Bergamaschi, Aleksia Kolo, Hyewon Jeong"
author = "Nassim Oufattole, Matthew McDermott, Teya Bergamaschi, Aleksia Kolo, Hyewon Jeong"
# Define the json_url for our version switcher.


json_url = "https://meds-tab.readthedocs.io/en/latest/_static/switcher.json"
# Define the version we use for matching in the version switcher.
version_match = os.environ.get("READTHEDOCS_VERSION")
release = MEDS_tabular_automl.__version__
# If READTHEDOCS_VERSION doesn't exist, we're not on RTD
# If it is an integer, we're in a PR build and the version isn't correct.
# If it's "latest" → change to "dev" (that's what we want the switcher to call it)
if not version_match or version_match.isdigit():
    # For local development, infer the version to match from the package.
    if "dev" in release or "rc" in release:
        version_match = "dev"
        # We want to keep the relative reference if we are in dev mode
        # but we want the whole url if we are effectively in a released version
        json_url = "_static/switcher.json"
    else:
        version_match = f"v{release}"
elif version_match == "latest":
    version_match = "dev"
elif version_match == "stable":
    version_match = f"v{release}"

# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

language = "en"
# -- Path setup

__location__ = Path(os.path.dirname(__file__))
__src__ = __location__ / "../.."

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
sys.path.insert(0, str(__src__))


def ensure_pandoc_installed(_):
    """Source: https://stackoverflow.com/questions/62398231/building-docs-fails-due-to-missing-pandoc"""
    import pypandoc

    # Download pandoc if necessary. If pandoc is already installed and on
    # the PATH, the installed version will be used. Otherwise, we will
    # download a copy of pandoc into docs/bin/ and add that to our PATH.
    pandoc_dir = str(__location__ / "bin")
    # Add dir containing pandoc binary to the PATH environment variable
    if pandoc_dir not in os.environ["PATH"].split(os.pathsep):
        os.environ["PATH"] += os.pathsep + pandoc_dir

    pypandoc.ensure_pandoc_installed(
        targetfolder=pandoc_dir,
        delete_installer=True,
    )


# -- Run sphinx-apidoc
# This ensures we don't need to run apidoc manually.

# TODO: use https://github.com/sphinx-extensions2/sphinx-autodoc2

from sphinx.ext import apidoc

output_dir = __location__ / "api"
module_dir = __src__ / "src/MEDS_tabular_automl"
if output_dir.is_dir():
    shutil.rmtree(output_dir)

try:
    cmd_line = f"--implicit-namespaces -e -f -o {output_dir} {module_dir}"
    apidoc.main(cmd_line.split(" "))
except Exception as e:  # pylint: disable=broad-except
    print(f"Running `sphinx-apidoc {cmd_line}` failed!\n{e}")


# -- General configuration


# -- Project information
extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx.ext.coverage",
    "sphinx.ext.ifconfig",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.imgconverter",
    "sphinxcontrib.collections",
    "sphinx_subfigure",
    "myst_parser",
    "nbsphinx",
    # "sphinx_immaterial",
]
# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "pydata_sphinx_theme"
# html_sidebars = {"**": []}  # ["logo-text.html", "globaltoc.html", "localtoc.html", "searchbox.html"]
html_sidebars = {
    "api/*": [
        "sidebar-nav-bs",
    ],
    "**": [],
}
nbsphinx_allow_errors = True


collections_dir = __location__ / "_collections"
if not collections_dir.is_dir():
    os.mkdir(collections_dir)

python_version = ".".join(map(str, sys.version_info[0:2]))
intersphinx_mapping = {
    "sphinx": ("https://www.sphinx-doc.org/en/master", None),
    "python": ("https://docs.python.org/" + python_version, None),
    "matplotlib": ("https://matplotlib.org", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "sklearn": ("https://scikit-learn.org/stable", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
    "pandera": ("https://pandera.readthedocs.io/en/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference", None),
    "setuptools": ("https://setuptools.pypa.io/en/stable/", None),
    "pyscaffold": ("https://pyscaffold.org/en/stable", None),
    "hyperimpute": ("https://hyperimpute.readthedocs.io/en/latest/", None),
    "xgbse": ("https://loft-br.github.io/xgboost-survival-embeddings/", None),
    "lifelines": ("https://lifelines.readthedocs.io/en/stable/", None),
    "optuna": ("https://optuna.readthedocs.io/en/stable/", None),
}
intersphinx_disabled_domains = ["std"]

templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Control options for included jupyter notebooks.
nb_execution_mode = "off"


# -- Options for HTML output

# Configure MyST-Parser
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_image",
    "linkify",
    "replacements",
    "smartquotes",
    "substitution",
    "tasklist",
]

myst_update_mathjax = True

# MyST URL schemes.
myst_url_schemes = {
    "http": None,
    "https": None,
    "ftp": None,
    "mailto": None,
    "repo-code": "https://github.com/mmcdermott/MEDS_Tabular_AutoML/tree/main/{{path}}#{{fragment}}",
    # "doi": "https://doi.org/{{path}}",
    # "gh-issue": {
    #     "url": "https://github.com/executablebooks/MyST-Parser/issue/{{path}}#{{fragment}}",
    #     "title": "Issue #{{path}}",
    #     "classes": ["github"],
    # },
}

# The suffix of source filenames.
source_suffix = [".rst", ".md"]

# The encoding of source files.
# source_encoding = 'utf-8-sig'

# The master toctree document.
master_doc = "index"

# The reST default role (used for this markup: `text`) to use for all documents.
default_role = "py:obj"

# If true, '()' will be appended to :func: etc. cross-reference text.
# add_function_parentheses = True

# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
# add_module_names = True

# If true, sectionauthor and moduleauthor directives will be shown in the
# output. They are ignored by default.
# show_authors = False

# The name of the Pygments (syntax highlighting) style to use.
# https://pygments.org/styles/
pygments_style = "tango"

# A list of ignored prefixes for module index sorting.
# modindex_common_prefix = []

# If true, keep warnings as "system message" paragraphs in the built documents.
# keep_warnings = False

# If this is True, todo emits a warning for each TODO entries. The default is False.
todo_emit_warnings = True


# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.


html_title = f"MEDS-Tab v{release} Documentation"
html_short_title = "MEDS-Tab"

# html_logo = "query-512.png"
# html_favicon = "query-16.ico"

# Material theme options (see theme.conf for more information)
html_theme_options = {
    "logo": {
        "text": "MEDS-TAB",
        "image_light": "../assets/dark_purple_meds_tab.png",
        "image_dark": "../assets/light_purple_meds_tab.png",
    },
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/mmcdermott/MEDS_Tabular_AutoML",  # required
            "icon": "fa-brands fa-github",
            "type": "fontawesome",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/meds-tab/",
            "icon": "fa-brands fa-python",
        },
    ],
    "header_links_before_dropdown": 6,
    "show_toc_level": 1,
    "navbar_align": "left",  # [left, content, right] For testing that the navbar items align properly
    # "show_nav_level": 2,
    "announcement": "This is a community-supported tool. If you'd like to contribute, <a href='https://github.com/mmcdermott/MEDS_Tabular_AutoML'>check out our GitHub repository.</a> Your contributions are welcome!",  # noqa E501
    "show_version_warning_banner": True,
    "switcher": {
        "json_url": json_url,
        "version_match": version_match,
    },
    "navbar_center": ["version-switcher", "navbar-nav"],
    "footer_start": ["copyright"],
    "footer_center": ["sphinx-version"],
    "use_edit_page_button": True,
    # "secondary_sidebar_items": {
    #     "**/*": ["page-toc", "edit-this-page", "sourcelink"],
    # },
    "back_to_top_button": True,
}

html_context = {
    "github_user": "mmcdermott",
    "github_repo": "MEDS_Tabular_AutoML",
    "github_version": "main",
    "doc_path": "docs/source",
}

# html_theme_options = {
#     # Set the name of the project to appear in the navigation.
#     "nav_title": "MEDS-TAB",
#     "palette": {"primary": "purple", "accent": "purple"},
#     # {
#     #     "media": "(prefers-color-scheme: light)",
#     #     "scheme": "default",
#     #     "toggle": {
#     #         "icon": "material/toggle-switch-off-outline",
#     #         "name": "Switch to dark mode",
#     #     },
#     # },
#     # {
#     #     "media": "(prefers-color-scheme: dark)",
#     #     "scheme": "slate",
#     #     "toggle": {
#     #         "icon": "material/toggle-switch",
#     #         "name": "Switch to light mode",
#     #     },
#     # },
#     # "color_primary": "green",
#     # "color_accent": "green",
#     # Set the repo location to get a badge with stats
#     "repo_url": "https://github.com/mmcdermott/MEDS_Tabular_AutoML",
#     "repo_name": "meds-tab",
#     # Visible levels of the global TOC; -1 means unlimited
#     "globaltoc_depth": 3,
#     # If False, expand all TOC entries
#     "globaltoc_collapse": True,
#     # If True, show hidden TOC entries
#     "globaltoc_includehidden": False,
# }


html_show_copyright = True
htmlhelp_basename = "meds-tab-doc"

# -- Options for LaTeX output
# latex_engine = "xelatex"
latex_elements = {  # type: ignore
    # The paper size ("letterpaper" or "a4paper").
    "papersize": "letterpaper",
    # The font size ("10pt", "11pt" or "12pt").
    "pointsize": "10pt",
    # Additional stuff for the LaTeX preamble.
    "preamble": "\n".join(
        [
            r"\usepackage{svg}",
            r"\DeclareUnicodeCharacter{2501}{-}",
            r"\DeclareUnicodeCharacter{2503}{|}",
            r"\DeclareUnicodeCharacter{2500}{-}",
            r"\DeclareUnicodeCharacter{2550}{-}",
            r"\DeclareUnicodeCharacter{2517}{+}",
            r"\DeclareUnicodeCharacter{2518}{+}",
            r"\DeclareUnicodeCharacter{2534}{+}",
            r"\DeclareUnicodeCharacter{250C}{+}",
            r"\DeclareUnicodeCharacter{252C}{+}",
            r"\DeclareUnicodeCharacter{2510}{+}",
            r"\DeclareUnicodeCharacter{2502}{|}",
            r"\DeclareUnicodeCharacter{2506}{|}",
            r"\DeclareUnicodeCharacter{2561}{|}",
            r"\DeclareUnicodeCharacter{256A}{|}",
            r"\DeclareUnicodeCharacter{2523}{|}",
            r"\DeclareUnicodeCharacter{03BC}{\ensuremath{\mu}}",
            r"\DeclareUnicodeCharacter{255E}{|}",
            r"\DeclareUnicodeCharacter{255F}{+}",
            r"\DeclareUnicodeCharacter{254E}{|}",
            r"\DeclareUnicodeCharacter{257C}{-}",
            r"\DeclareUnicodeCharacter{257E}{-}",
            r"\DeclareUnicodeCharacter{2559}{+}",
        ]
    ),
}


# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title, author, documentclass [howto/manual]).
latex_documents = [
    (
        "index",
        "meds_tab_documentation.tex",
        "MEDS-TAB Documentation",
        r"Matthew McDermott \& Nassim Oufattole \& Teya Bergamaschi",
        "manual",
    )
]

# The name of an image file (relative to this directory) to place at the top of
# the title page.
# latex_logo = ""

# For "manual" documents, if this is true, then toplevel headings are parts,
# not chapters.
# latex_use_parts = False

# -- Options for EPUB output
epub_show_urls = "footnote"

print(f"loading configurations for {project} {release} ...", file=sys.stderr)


def setup(app):
    app.connect("builder-inited", ensure_pandoc_installed)
