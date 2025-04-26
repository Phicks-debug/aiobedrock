from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="aiobedrock",
    version="0.1.0",
    author="Phicks",
    author_email="an.tq@techxcorp.com",
    description="AWS boto3 bedrock client in async",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Phicks-debug/aiobedrock",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3.14",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: System :: Logging",
    ],
    python_requires=">=3.9",
)
