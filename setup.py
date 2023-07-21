import pathlib

from setuptools import find_packages, setup

here = pathlib.Path(__file__).parent.resolve()
long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name="Housing_price",
    version="0.0.1",
    description="Housing Price Prediction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pranathiparuchu/mle-training_1",
    author="Pranathi",
    author_email="pranathi.paruchu@tigeranalytics.com",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8, <4",
    project_urls={"Source": "https://github.com/pranathiparuchu/mle-training_1"},
)