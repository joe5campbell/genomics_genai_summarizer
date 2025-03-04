#!/usr/bin/env python
"""Script to convert the fine-tuned model to ONNX format."""

import os
import logging
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
MODEL_DIR = os.path.join("models", "checkpoints", "genomics_summarizer_final")
ONNX_DIR = os.path.join("models", "onnx")
ONNX_MODEL_PATH = os.path.join(ONNX_DIR, "model.onnx")
TOKENIZER_DIR = os.path.join("data", "processed", "tokenizer")  # Use the tokenizer we saved during preprocessing
MAX_LENGTH = 256  # Match the value used in training

def convert_to_onnx(model_path, onnx_path, tokenizer_path, max_length):
    """
    Convert the PyTorch model to ONNX format.
    Args:
        model_path (str): Path to the fine-tuned model
        onnx_path (str): Path to save the ONNX model
        tokenizer_path (str): Path to the tokenizer
        max_length (int): Maximum sequence length
    """
    logger.info(f"Converting model at {model_path} to ONNX format")
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    # Set the model to evaluation mode
    model.eval()
    
    # Create dummy input
    dummy_input = tokenizer(
        "summarize: This is a dummy input text to convert the model to ONNX format.", 
        return_tensors="pt",
        padding="max_length",
        max_length=max_length,
        truncation=True
    )
    
    # Set up symbolic names for inputs and outputs
    input_names = ["input_ids", "attention_mask"]
    output_names = ["logits"]
    
    # Dynamic axes for variable batch size and sequence length
    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
        "logits": {0: "batch_size", 1: "sequence_length"}
    }
    
    # Export the model to ONNX
    logger.info("Exporting model to ONNX...")
    with torch.no_grad():
        torch.onnx.export(
            model,
            (dummy_input["input_ids"], dummy_input["attention_mask"]),
            onnx_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=11,
            do_constant_folding=True,
            verbose=False
        )
    
    logger.info(f"Model successfully exported to {onnx_path}")

def validate_onnx_model(onnx_path):
    """
    Validate the exported ONNX model.
    Args:
        onnx_path (str): Path to the ONNX model
    """
    try:
        import onnx
        
        # Load the ONNX model
        logger.info(f"Validating ONNX model at {onnx_path}")
        onnx_model = onnx.load(onnx_path)
        
        # Check the model for errors
        onnx.checker.check_model(onnx_model)
        logger.info("ONNX model validation successful!")
        
        # Print model metadata
        logger.info(f"Model IR version: {onnx_model.ir_version}")
        logger.info(f"Opset version: {onnx_model.opset_import[0].version}")
        
        return True
    except ImportError:
        logger.warning("ONNX package not installed. Skipping validation.")
        return False
    except Exception as e:
        logger.error(f"ONNX model validation failed: {str(e)}")
        return False

def verify_onnx_runtime(onnx_path, tokenizer_path):
    """
    Verify the ONNX model with ONNX Runtime.
    Args:
        onnx_path (str): Path to the ONNX model
        tokenizer_path (str): Path to the tokenizer
    """
    try:
        import onnxruntime as ort
        
        logger.info("Testing inference with ONNX Runtime")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        
        # Create an ONNX Runtime session
        session_options = ort.SessionOptions()
        session = ort.InferenceSession(onnx_path, session_options)
        
        # Prepare input
        test_text = "summarize: This is a test input for the ONNX model."
        inputs = tokenizer(
            test_text,
            return_tensors="pt",
            padding="max_length",
            max_length=MAX_LENGTH,
            truncation=True
        )
        
        # Run inference
        ort_inputs = {
            "input_ids": inputs["input_ids"].numpy(),
            "attention_mask": inputs["attention_mask"].numpy()
        }
        
        ort_outputs = session.run(None, ort_inputs)
        
        logger.info("ONNX Runtime inference successful!")
        
        return True
    except ImportError:
        logger.warning("ONNX Runtime not installed. Skipping runtime verification.")
        return False
    except Exception as e:
        logger.error(f"ONNX Runtime verification failed: {str(e)}")
        return False

def main(args):
    """Main function to optimize the model."""
    model_path = args.model_dir
    tokenizer_path = args.tokenizer_dir
    onnx_dir = args.onnx_dir
    max_length = args.max_length
    
    # Create ONNX directory if it doesn't exist
    os.makedirs(onnx_dir, exist_ok=True)
    
    # Define ONNX model path
    onnx_path = os.path.join(onnx_dir, "model.onnx")
    
    # Convert the model to ONNX
    convert_to_onnx(model_path, onnx_path, tokenizer_path, max_length)
    
    # Validate the ONNX model
    valid = validate_onnx_model(onnx_path)
    
    if valid:
        # Verify the ONNX model with ONNX Runtime
        verify_onnx_runtime(onnx_path, tokenizer_path)
    
    # Copy tokenizer files to ONNX directory for convenience
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.save_pretrained(onnx_dir)
    logger.info(f"Saved tokenizer to {onnx_dir}")
    
    logger.info("Model optimization complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert fine-tuned model to ONNX format")
    parser.add_argument("--model_dir", type=str, default=MODEL_DIR, help="Directory containing the fine-tuned model")
    parser.add_argument("--tokenizer_dir", type=str, default=TOKENIZER_DIR, help="Directory containing the tokenizer")
    parser.add_argument("--onnx_dir", type=str, default=ONNX_DIR, help="Directory to save the ONNX model")
    parser.add_argument("--max_length", type=int, default=MAX_LENGTH, help="Maximum sequence length")
    
    args = parser.parse_args()
    main(args)