from pathlib import Path
from setuptools import find_namespace_packages, setup

BASE_DIR = Path(__file__).parent

extra_packages = [
    "pylama",
    "isort",
    "autopep8"
]



setup(
    name='ocular_predictor',
    packages=find_namespace_packages(),
    version='0.0.1',
    description='Create a temporal fixation predictor, to predict when the eye moves',
    author="Camilo Meneses Zambeat",
    python_requires=">=3.8",
    extras_require={"dev": extra_packages}
)
