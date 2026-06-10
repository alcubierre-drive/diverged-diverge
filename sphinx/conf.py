import os
import sys
import subprocess
import datetime

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'divERGe'
copyright = f'{datetime.date.today().year}, Jonas B. Profe, Lennart Klebl'
author = 'Jonas B. Profe, Lennart Klebl'

# getting some release info
_tag = subprocess.check_output( ['bash', '-c', 'git tag --list "v*.*" | tail -n1'] )
_commit = subprocess.check_output( ['bash', '-c', 'git rev-parse --short HEAD'] )
_dirty = subprocess.check_output( ['bash', '-c', "(git diff --quiet && git diff --cached --exit-code --quiet) || echo ':dirty'"] )
_branch = subprocess.check_output( ['bash', '-c', 'git rev-parse --abbrev-ref HEAD'] )

release = _tag.decode('utf8')[1:-1]
branch = _branch.decode('utf8')[:-1]
if branch == 'pages':
    branch = 'master'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["hawkmoth", "sphinx.ext.autodoc", "sphinx.ext.extlinks"]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_logo = "img/diverge.png"
html_static_path = ['_static',]
html_css_files = ["custom.css",]
html_theme_options = { "logo_only": True, }

# C doc config
hawkmoth_root = os.path.abspath('../src')
hawkmoth_source_uri = 'https://codeberg.org/lennart_klebl/divERGe/src/branch/%s/src/{source}#L{line}' % branch

# links
extlinks = {
        'release_version': ('https://codeberg.org/lennart_klebl/divERGe/raw/branch/' + branch + '/public/releases/v' + release + '/%s', None),
        'release_tree': ('https://codeberg.org/lennart_klebl/divERGe/src/branch/' + branch + '/%s', None),
}
extlinks_detect_hardcoded_links = True

# is this needed?
sys.path.append( os.path.abspath('../util') )

# and here the python modules for documentation
import diverge
import diverge.output
