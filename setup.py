#!/usr/bin/env python3
"""
Setup script for NHTSA Vehicle Data Enrichment Pipeline
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file for long description
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Read requirements from requirements.txt
requirements = []
requirements_file = this_directory / "requirements.txt"
if requirements_file.exists():
    with open(requirements_file, 'r', encoding='utf-8') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="nhtsa-vehicle-enrichment",
    version="1.0.0",
    author="Data Engineering Team",
    author_email="your.email@example.com",
    description="NHTSA Vehicle Data Enrichment Pipeline for adding missing cylinder data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/denisigorevich/Data-Engineering-Capstone-Project",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    package_data={
        "data_enrichment": ["*.yaml", "*.yml"],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "responses>=0.23.0",
            "ruff>=0.1.0",
            "pre-commit>=3.0.0",
        ],
        "test": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "responses>=0.23.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "nhtsa-enrich=src.data_enrichment.enrich_with_cylinders:main",
            "nhtsa-batch-enrich=src.data_enrichment.batch_enrich:main",
        ],
    },
    keywords="nhtsa, vehicle, data, enrichment, api, automotive",
    project_urls={
        "Bug Reports": "https://github.com/denisigorevich/Data-Engineering-Capstone-Project/issues",
        "Source": "https://github.com/denisigorevich/Data-Engineering-Capstone-Project",
        "Documentation": "https://github.com/denisigorevich/Data-Engineering-Capstone-Project/blob/main/README.md",
    },
) 
