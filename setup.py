#!/usr/bin/env python
"""
Setup script to create the initial directory structure for the genomics text summarization project.
"""
import os

def create_project_structure():
    """Create the necessary directories for the project."""
    # Define the directories to create
    directories = [
        'src',
        'data',
        'data/raw',
        'data/processed',
        'models',
        'models/checkpoints',
        'models/onnx'
    ]
    
    # Create each directory if it doesn't exist
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")
    
    # Create empty __init__.py files in the src directory for proper imports
    with open(os.path.join('src', '__init__.py'), 'w') as f:
        pass
    
    # Create the basic README.md file
    with open('README.md', 'w') as f:
        f.write("""# Genomics Text Summarization

A project to fine-tune a small-scale generative AI model to summarize genomic research data.

## Project Overview

This project fine-tunes a small language model (like Llama-2-3B or GPT-2) to generate concise summaries of genomics research papers and data. The model is optimized for efficient deployment and inference.

## Features

- Downloads and preprocesses genomics text data
- Fine-tunes a pretrained language model for summarization
- Converts the model to ONNX format for efficient inference
- Provides a FastAPI-based API for generating summaries
- Deploys the model to AzureML
- Offers a Streamlit web interface

## Getting Started

See the documentation for instructions on setup and usage.
""")
    
    print("Created basic README.md")
    
    # List of Python scripts to create
    scripts = {
        'preprocess.py': '"""Data preprocessing script for the genomics text summarization project."""',
        'train.py': '"""Training script for fine-tuning the language model."""',
        'optimize.py': '"""Script to convert the fine-tuned model to ONNX format."""',
        'inference.py': '"""Inference script to generate summaries from genomic text."""',
        'deploy.py': '"""FastAPI deployment script."""',
        'azure_deploy.py': '"""Script for deploying the model to AzureML."""',
        'app.py': '"""Streamlit app for the frontend interface."""'
    }
    
    # Create each script with a basic docstring
    for script_name, docstring in scripts.items():
        script_path = os.path.join('src', script_name)
        with open(script_path, 'w') as f:
            f.write(f"""#!/usr/bin/env python
{docstring}

import os
import sys

def main():
    \"\"\"Main function.\"\"\"
    print(f"Running {os.path.basename(__file__)}")
    
if __name__ == "__main__":
    main()
""")
        print(f"Created script: {script_path}")
    
    # Create a basic Dockerfile
    with open('Dockerfile', 'w') as f:
        f.write("""FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "src.deploy:app", "--host", "0.0.0.0", "--port", "8000"]
""")
    
    print("Created Dockerfile")
    
    print("\nProject structure setup complete!")

if __name__ == "__main__":
    create_project_structure()