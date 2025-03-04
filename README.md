# Genomics Text Summarization

This project implements a fine-tuned generative AI model to summarize genomic research data. The model is trained on genomics-related text data and optimized to produce concise, readable summaries of complex genomic research findings, with focus on low-cost implementation and deployment.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Directory Structure](#directory-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Preprocessing Data](#preprocessing-data)
  - [Training the Model](#training-the-model)
  - [Optimizing the Model](#optimizing-the-model)
  - [Running Inference](#running-inference)
  - [API Deployment](#api-deployment)
  - [AzureML Deployment](#azureml-deployment)
  - [Streamlit Frontend](#streamlit-frontend)
- [Deployment Workflow](#deployment-workflow)
- [Technical Details](#technical-details)
- [License](#license)

## Project Overview

This project fine-tunes a small-scale language model (GPT-2) to generate concise summaries of genomics research data. It focuses on:

1. **Data Processing**: Downloading and preprocessing genomics text data from public datasets
2. **Model Fine-tuning**: Training the model on a specific task (summarization) with domain-specific data
3. **Model Optimization**: Converting the model to ONNX format for faster inference
4. **API Development**: Creating a FastAPI-based REST API to serve the model
5. **AzureML Deployment**: Deploying the model to Azure Machine Learning for scalable inference
6. **User Interface**: Building a Streamlit-based frontend for easy interaction

## Features

- 📊 Downloads and preprocesses genomics text data from Hugging Face datasets
- 🧠 Fine-tunes a pretrained language model (GPT-2) for domain-specific summarization
- ⚡ Optimizes inference speed using ONNX runtime
- 🔄 Creates a reproducible ML pipeline from preprocessing to deployment
- 🌐 Provides a REST API for serving model predictions
- ☁️ Deploys efficiently to AzureML using Azure Container Instances
- 🖥️ Offers a user-friendly web interface built with Streamlit

## Directory Structure

```
genomics-text-summarization/
├── src/                           # Main source code
│   ├── preprocess.py              # Data preprocessing
│   ├── train.py                   # Model training
│   ├── optimize.py                # Model optimization (ONNX conversion)
│   ├── inference.py               # Inference utilities
│   ├── deploy.py                  # FastAPI deployment
│   ├── azure_deploy.py            # AzureML deployment
│   ├── app.py                     # Streamlit web app
├── data/                          # Data storage
│   ├── raw/                       # Raw datasets
│   ├── processed/                 # Processed datasets
├── models/                        # Model storage
│   ├── checkpoints/               # Trained PyTorch models
│   ├── onnx/                      # ONNX models
├── deployment/                    # Deployment files for AzureML
├── Dockerfile                     # Docker container definition
├── requirements.txt               # Python dependencies
├── README.md                      # Project documentation
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/genomics-text-summarization.git
   cd genomics-text-summarization
   ```

2. Create a virtual environment:
   ```bash
   python -m venv ggs_venv
   source ggs_venv/bin/activate  # On Windows, use: ggs_venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up the project structure:
   ```bash
   python setup.py
   ```

## Usage

### Preprocessing Data

Download and preprocess the genomics text data:

```bash
python src/preprocess.py
```

This will:
- Download a genomics-related dataset from Hugging Face
- Clean and normalize the text
- Extract text and summary pairs
- Split into training and validation sets
- Tokenize the data
- Save processed data for training

### Training the Model

Fine-tune the model on the preprocessed data:

```bash
python src/train.py --model gpt2 --batch_size 4 --epochs 3 --lr 5e-5
```

Options:
- `--model`: Model name (default: gpt2)
- `--batch_size`: Batch size for training (default: 4)
- `--epochs`: Number of training epochs (default: 3)
- `--lr`: Learning rate (default: 5e-5)

The trained model will be saved to `models/checkpoints/genomics_summarizer_final`.

### Optimizing the Model

Convert the PyTorch model to ONNX format for faster inference:

```bash
python src/optimize.py --model_dir models/checkpoints/genomics_summarizer_final
```

Options:
- `--model_dir`: Directory containing the fine-tuned model
- `--onnx_dir`: Directory to save the ONNX model (default: models/onnx)
- `--max_length`: Maximum sequence length (default: 512)

The ONNX model will be saved to `models/onnx/model.onnx`.

### Running Inference

Generate summaries from genomic text:

```bash
python src/inference.py --text "Your genomic text here..." --use_onnx
```

or using a file:

```bash
python src/inference.py --input_file path/to/genomic_text.txt --output_file path/to/summary.txt --use_onnx
```

Options:
- `--model_path`: Path to the fine-tuned model (default: models/checkpoints/genomics_summarizer_final)
- `--onnx_path`: Path to the ONNX model (default: models/onnx/model.onnx)
- `--use_onnx`: Whether to use ONNX for inference (default: True)
- `--text`: Input text to summarize
- `--input_file`: Path to input text file
- `--output_file`: Path to save the summary
- `--max_new_tokens`: Maximum number of new tokens to generate (default: 100)

### API Deployment

Run the FastAPI server locally:

```bash
uvicorn src.deploy:app --host 0.0.0.0 --port 8000
```

This will start the API server at http://localhost:8000, with the following endpoints:
- `GET /`: Root endpoint with basic info
- `GET /health`: Health check endpoint
- `POST /summarize`: Summarize genomic text
- `GET /models`: Get information about available models

Example API call:
```bash
curl -X POST "http://localhost:8000/summarize" \
     -H "Content-Type: application/json" \
     -d '{"text": "Your genomic text here...", "max_new_tokens": 100}'
```

### AzureML Deployment

Deploy the model to AzureML:

```bash
python src/azure_deploy.py \
    --use_onnx \
    --register \
    --deploy \
    --subscription_id "your-subscription-id" \
    --resource_group "your-resource-group" \
    --workspace_name "your-workspace-name"
```

Options:
- `--model_dir`: Path to the PyTorch model (default: models/checkpoints/genomics_summarizer_final)
- `--onnx_model_path`: Path to the ONNX model (default: models/onnx/model.onnx)
- `--deployment_dir`: Directory to store deployment files (default: deployment)
- `--use_onnx`: Whether to deploy the ONNX model (default: True)
- `--register`: Register the model to AzureML
- `--deploy`: Deploy the model to an endpoint
- `--subscription_id`: Azure subscription ID
- `--resource_group`: Azure resource group
- `--workspace_name`: AzureML workspace name

### Streamlit Frontend

Run the Streamlit web app:

```bash
streamlit run src/app.py
```

Set the API URL in the app if using AzureML:
```bash
API_URL=https://your-endpoint.azureml.net streamlit run src/app.py
```

This will start the web interface at http://localhost:8501.

## Deployment Workflow

The full deployment workflow consists of these steps:

1. **Preprocess the data**: `python src/preprocess.py`
2. **Train the model**: `python src/train.py`
3. **Optimize the model**: `python src/optimize.py`
4. **Deploy to AzureML**: `python src/azure_deploy.py --register --deploy --use_onnx`
5. **Run the web app**: `streamlit run src/app.py`

## Technical Details

- **Model**: GPT-2 (small, 117M parameters)
- **Training Data**: PubMed abstracts or ArXiv papers (automatically downloaded)
- **Optimization**: ONNX Runtime for faster inference
- **API**: FastAPI
- **Deployment**: AzureML & Azure Container Instances
- **Frontend**: Streamlit
- **Containerization**: Docker

## License

This project is released under the MIT License.