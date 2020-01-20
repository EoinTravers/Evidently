import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open('requirements.txt') as f:
    required = f.read().splitlines()

setuptools.setup(
    name="evidently",
    version="0.0.1-alpha",
    author="Eoin Travers",
    author_email="eoin.travers@gmail.com",
    description="Efficient simulation of evidence accumulation models.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/eointravers/evidently",
    packages=setuptools.find_packages(),
    classifiers=[
    ],
    install_reqs = required
)
