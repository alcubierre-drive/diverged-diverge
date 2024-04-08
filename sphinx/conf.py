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
copyright = f'{datetime.date.today().year}, Lennart Klebl, Jonas B. Profe'
author = 'Lennart Klebl, Jonas B. Profe'

# getting some release info
_tag = subprocess.check_output( ['bash', '-c', 'git tag --list "v*.*" | tail -n1'] )
_commit = subprocess.check_output( ['bash', '-c', 'git rev-parse --short HEAD'] )
_dirty = subprocess.check_output( ['bash', '-c', "(git diff --quiet && git diff --cached --exit-code --quiet) || echo ':dirty'"] )
_branch = subprocess.check_output( ['bash', '-c', 'git rev-parse --abbrev-ref HEAD'] )

release = _tag.decode('utf8')[1:-1]
branch = _branch.decode('utf8')[:-1]

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["hawkmoth", "sphinx.ext.autodoc", "sphinx.ext.extlinks"]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']

# C doc config
hawkmoth_root = os.path.abspath('../src')
hawkmoth_source_uri = 'https://git.rwth-aachen.de/frg/diverge/-/blob/%s/src/{source}#L{line}' % branch

# links
extlinks = {
        'release_version': ('https://git.rwth-aachen.de/frg/diverge/-/raw/' + branch + '/public/releases/v' + release + '/%s', None),
        #'release_raw': ('https://git.rwth-aachen.de/frg/diverge/-/raw/' + branch + '/%s', None),
        'release_tree': ('https://git.rwth-aachen.de/frg/diverge/-/tree/' + branch + '/%s', None)
}
extlinks_detect_hardcoded_links = True

# is this needed?
sys.path.append( os.path.abspath('../util') )

# and here the python modules for documentation
import diverge
import diverge.output
