#!/usr/bin/env python3
"""
Setup script for Virtual Tutor - AI-Powered Educational Assistant
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="virtual-teaching-assistant-using-rag",
    version="1.0.0",
    author="Virtual Tutor Team",
    author_email="contact@virtualtutor.com",
    description="RAG-Powered Virtual Teaching Assistant with multi-modal input support",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/virtual-teaching-assistant-using-rag",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Education",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=8.0.2",
            "pytest-cov>=4.1.0",
            "black>=24.2.0",
            "flake8>=7.0.0",
            "mypy>=1.8.0",
        ],
        "gpu": [
            "torch>=2.5.1",
            "torchvision>=0.20.1",
            "faiss-gpu>=1.9.0",
        ],
        "full": [
            "opencv-python>=4.9.0.80",
            "pdfplumber>=0.10.3",
            "openpyxl>=3.1.2",
            "aiohttp>=3.9.3",
            "redis>=5.0.1",
        ],
    },
    entry_points={
        "console_scripts": [
            "virtual-teaching-assistant=app:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.json", "*.html", "*.css", "*.js"],
    },
    keywords="education, ai, rag, teaching-assistant, chatbot, nlp, speech-recognition, ocr, virtual-tutor",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/virtual-teaching-assistant-using-rag/issues",
        "Source": "https://github.com/yourusername/virtual-teaching-assistant-using-rag",
        "Documentation": "https://github.com/yourusername/virtual-teaching-assistant-using-rag#readme",
    },
) 