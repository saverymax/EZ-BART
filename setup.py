import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="EZ-BART",
    version="1.0.0",
    author="Max Savery Author",
    author_email="max.savery@nih.gov",
    description="Tool for BART summarization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/saverymax/EZ-BART",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
