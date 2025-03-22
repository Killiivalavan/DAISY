"""
Setup script for the DAISY voice assistant.
"""
from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="daisy-assistant",
    version="0.1.0",
    description="DAISY - Digital AI System for You - Voice Assistant",
    author="DAISY Development Team",
    packages=find_packages(),
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "daisy=daisy:main",
        ],
    },
    python_requires=">=3.8",
) 