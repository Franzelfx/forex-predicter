import os
import sys
from sphinx.highlighting import lexers
from pygments.lexers.python import PythonLexer
from sphinx.highlighting import PygmentsBridge

sys.path.insert(0, os.path.abspath('../../'))

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'forex-predicter'
copyright = '2023, Fabian Franz'
author = 'Fabian Franz'
release = '2023.06.02'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc', 'sphinx.ext.viewcode', 'm2r', 'numpydoc']

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', 'test']


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_css_files = ['custom.css']

# Generate the .rst files automatically from the Python modules
# Run the following command to generate/update the documentation:
#   sphinx-apidoc -o <output-dir> <source-dir> --force
# This command will generate the .rst files in the specified output directory
# based on the Python modules found in the source directory.
# You can then include the generated .rst files in your documentation.
# For example, to generate the .rst files in the 'modules' directory:
#   sphinx-apidoc -o modules ../../src --force
# The '--force' option ensures that the .rst files are always updated.
def run_apidoc(_):
    from sphinx.ext.apidoc import main
    output_path = os.path.join(os.path.dirname(__file__), '_modules')
    modules_path = os.path.abspath('../../src')
    main(['-e', '-M', '-o', output_path, modules_path])

def add_custom_css(app):
    app.add_css_file('custom.css')

# Register the setup function to be called when building the documentation
def setup(app):
    app.connect('builder-inited', run_apidoc)
    app.connect('builder-inited', add_custom_css)
    app.add_role('highlight', PygmentsBridge())
    app.add_css_file('custom.css')
    app.connect('builder-inited', add_custom_css)
    app.add_css_file('custom.css')