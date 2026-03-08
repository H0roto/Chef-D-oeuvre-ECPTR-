import os
from setuptools import setup, find_packages

# Lecture automatique du requirements.txt
with open('requirements.txt') as f:
    required = [
        line.strip() 
        for line in f 
        if line.strip() and not line.startswith('#')
    ]

setup(
    name="ulm3d",
    version="1.2",
    description="3D ULM",
    long_description="Volumetric Ultrasound Localization Microscopy",
    long_description_content_type="text/markdown",
    author="LIB",
    license="CC BY-NC-SA 4.0",
    url="https://github.com/Lab-Imag-Bio",
    packages=setuptools.find_packages(
        exclude=["*.tests", "*.tests.*", "tests.*", "tests", "scripts"],
    ),
    classifiers=[
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Topic :: Software Development :: Embedded Systems",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    install_requires=required,
    python_requires=">=3.10", 
)
