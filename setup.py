from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, "README.rst"), encoding="utf-8") as f:
    long_description = f.read()

setup(name="moonz",
      version="0.0.2",
      description="Automatic spectroscopic redshifting for MOONS",
      long_description=long_description,
      url="https://moonz.readthedocs.io",
      author="Adam Carnall",
      author_email="adamc@roe.ac.uk",
      packages=["moonz"],
      include_package_data=True,
      install_requires=["numpy", "deepdish", "sklearn", "astropy", "pandas",
                        "bagpipes", "spectres"],
      project_urls={"readthedocs": "https://moonz.readthedocs.io",
                    "GitHub": "https://github.com/ACCarnall/moonz"})
