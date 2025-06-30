from setuptools import setup, find_packages

setup(
    name="uccl",
    version="0.0.1",
    author="Yang Zhou",
    author_email="yangzhou.rpc@gmail.com",
    description="UCCL: Ultra and Unified CCL",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/uccl-project/uccl",
    packages=find_packages(),
    install_requires=[
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
    ],
    python_requires='>=3.8',
)