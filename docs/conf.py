import os
import sys

sys.path.insert(0, os.path.abspath('..'))
import moonz

extensions = ['sphinx.ext.autodoc', 'sphinx.ext.napoleon', 'sphinx_navtree']
source_suffix = '.rst'
master_doc = 'index'
project = u'Moonz'
copyright = u'2023, Adam Carnall'
author = u'Adam Carnall'
version = "0.0.2"
release = "0.0.2"
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
pygments_style = 'sphinx'
navtree_maxdepth = {'default': 1}
html_theme = 'sphinx_rtd_theme'
