import os
import sys
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

extensions = ['sphinx.ext.autodoc', 'sphinx.ext.napoleon', 'sphinx.ext.viewcode', 'myst_parser']

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



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
    modules_path = os.path.abspath('../../')
    main(['-e', '-M', '-o', output_path, modules_path])


def setup(app):
    app.connect('builder-inited', run_apidoc)