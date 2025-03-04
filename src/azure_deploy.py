#!/usr/bin/env python
"""Script for deploying the model to AzureML."""

import os
import argparse
import logging
import json
import shutil
from pathlib import Path
from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    Model,
    Environment,
    CodeConfiguration
)
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
MODEL_DIR = os.path.join("models", "checkpoints", "genomics_summarizer_final")
ONNX_MODEL_PATH = os.path.join("models", "onnx", "model.onnx")
DEPLOYMENT_DIR = "deployment"
DEPLOYMENT_SRC_DIR = os.path.join(DEPLOYMENT_DIR, "src")

# Default Azure settings
DEFAULT_SUBSCRIPTION_ID = "689e0ec9-f84a-4823-98d8-6a1d73880df8"
DEFAULT_RESOURCE_GROUP = "genomics-summarizer-rg"
DEFAULT_WORKSPACE_NAME = "genomics-summarizer-ws"
DEFAULT_ENDPOINT_NAME = "genomics-summarizer-endpoint"
DEFAULT_DEPLOYMENT_NAME = "genomics-summarizer-deployment"
DEFAULT_REGISTER = True
DEFAULT_DEPLOY = True

def setup_deployment_files(model_dir, onnx_model_path, deployment_dir, use_onnx=True):
    """
    Set up the deployment files.
    Args:
        model_dir (str): Path to the PyTorch model
        onnx_model_path (str): Path to the ONNX model
        deployment_dir (str): Directory to store deployment files
        use_onnx (bool): Whether to deploy the ONNX model
    """
    logger.info(f"Setting up deployment files in {deployment_dir}")
    os.makedirs(deployment_dir, exist_ok=True)
    
    # Create scoring script
    scoring_script_path = os.path.join(deployment_dir, "score.py")
    with open(scoring_script_path, "w") as f:
        f.write("""import json\ndef init(): pass\ndef run(data): return json.dumps({"message": "Hello from deployment"})""")
    logger.info(f"Scoring script created at {scoring_script_path}")

    # Create environment file
    conda_env_path = os.path.join(deployment_dir, "conda_env.yml")
    with open(conda_env_path, "w") as f:
        f.write("""name: deployment-env\ndependencies:\n  - python=3.8\n  - pip:\n    - azureml-defaults\n    - onnxruntime""")
    logger.info(f"Conda environment file created at {conda_env_path}")

    return scoring_script_path, conda_env_path

def register_model(ml_client, model_path):
    """Registers the model to AzureML."""
    model = Model(
        path=model_path,
        name="genomics-summarizer-model",
        type=AssetTypes.CUSTOM_MODEL,
        version=None,
    )
    registered_model = ml_client.models.create_or_update(model)
    logger.info(f"Model registered with name: {registered_model.name} and version: {registered_model.version}")
    return registered_model

def deploy_model(ml_client, registered_model, endpoint_name, deployment_name, scoring_script, env_file):
    """Deploys the registered model to AzureML endpoint."""
    # Ensure endpoint exists
    try:
        # Try to get existing endpoint
        endpoint = ml_client.online_endpoints.get(endpoint_name)
        logger.info(f"Using existing endpoint: {endpoint_name}")
    except:
        # Create new endpoint if it doesn't exist
        logger.info(f"Creating new endpoint: {endpoint_name}")
        endpoint = ManagedOnlineEndpoint(name=endpoint_name)
        ml_client.online_endpoints.begin_create_or_update(endpoint).wait()
        logger.info(f"Created new endpoint: {endpoint_name}")
    
    # Define environment
    environment = Environment(
        name="genomics-summarizer-env",
        conda_file=env_file,
        image="mcr.microsoft.com/azureml/curated/minimal-ubuntu20.04-py38-cpu:latest"
    )
    
    # Define deployment
    deployment = ManagedOnlineDeployment(
        name=deployment_name,
        endpoint_name=endpoint_name,
        model=registered_model,
        environment=environment,
        code_configuration=CodeConfiguration(
            code=os.path.dirname(scoring_script), 
            scoring_script=os.path.basename(scoring_script)
        ),
        instance_type="Standard_DS3_v2",
        instance_count=1,
    )
    
    # Create or update deployment
    ml_client.online_deployments.begin_create_or_update(deployment).wait()
    logger.info(f"Deployment {deployment_name} created at endpoint {endpoint_name}")
    
    # Update traffic to use this deployment
    endpoint = ml_client.online_endpoints.get(endpoint_name)
    endpoint.traffic = {deployment_name: 100}
    ml_client.online_endpoints.begin_create_or_update(endpoint).wait()
    logger.info(f"Updated endpoint traffic to 100% for deployment: {deployment_name}")

def main():
    """Main function to handle model registration and deployment."""
    credential = DefaultAzureCredential()
    ml_client = MLClient(credential, DEFAULT_SUBSCRIPTION_ID, DEFAULT_RESOURCE_GROUP, DEFAULT_WORKSPACE_NAME)
    
    scoring_script, env_file = setup_deployment_files(MODEL_DIR, ONNX_MODEL_PATH, DEPLOYMENT_DIR)
    registered_model = register_model(ml_client, ONNX_MODEL_PATH)
    deploy_model(ml_client, registered_model, DEFAULT_ENDPOINT_NAME, DEFAULT_DEPLOYMENT_NAME, scoring_script, env_file)

if __name__ == "__main__":
    main()