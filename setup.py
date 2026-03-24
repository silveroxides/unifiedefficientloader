import os
from setuptools import setup, find_packages, Extension

# Read the contents of the README file
with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

# Placeholder for future C/C++ extensions
# e.g. extensions = [Extension("unifiedefficientloader._ext", sources=["src/ext.cpp"])]
extensions = []

setup(
    name="unifiedefficientloader",
    version="0.1.0",
    author="Author",
    description="A unified interface for loading safetensors, handling CPU/GPU pinned transfers, and converting between tensors and dicts.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    extras_require={
        "all": ["torch", "safetensors", "tqdm"],
    },
    ext_modules=extensions,
)