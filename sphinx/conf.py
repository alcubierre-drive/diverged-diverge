# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'divERGe'
copyright = '2024, Lennart Klebl, Jonas B. Profe'
author = 'Lennart Klebl, Jonas B. Profe'
release = '0.6'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["hawkmoth", "sphinx.ext.autodoc"]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']

import os
hawkmoth_root = os.path.abspath('../src')
hawkmoth_source_uri = 'https://git.rwth-aachen.de/frg/diverge/-/blob/master/src/{source}#L{line}'

import sys
sys.path.append( os.path.abspath('../util') )

import diverge
import diverge.output
