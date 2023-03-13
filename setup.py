from setuptools import setup, find_packages

setup(
    name="misusing-llms",
    version="0.1",
    author="Tobias Deusser",
    packages=find_packages(),
    install_requires=[
        "torch",
        "fluidml==0.3.1",
        "datasets",
        "tokenizers",
        "pytorch-lightning",
        "torch",
        "tqdm",
        "transformers",
        "wandb",
        "pyyaml",
        "rich",
        "numpy",
        "markdown",
    ],
)
