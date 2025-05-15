# setup.py
from setuptools import setup, find_packages
from datetime import datetime

version = datetime.now().strftime("%Y.%m.%d.%H.%M")

setup(
    name="lightning_trainer_utils",
    version=version,
    author="Manav Mahan Singh",
    author_email="manav@genaec.ai",
    description="A Python package for using PyTorch Lightning with custom callbacks and model wrappers.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/manavmahan/lightning-trainer-utils",
    packages=find_packages(),
    python_requires=">=3.11",
)
