#!/usr/bin/env python
"""Script to convert the fine-tuned model to ONNX format for faster inference."""

import os
import logging
import argparse
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
MODEL_DIR = os.path.join("models", "checkpoints", "genomics_summarizer_final")
ONNX_DIR = os.path.join("models", "onnx")
MAX_LENGTH = 512  # Maximum sequence length for ONNX model


def convert_to_onnx(model_dir, onnx_dir, max_length=MAX_LENGTH):
    """
    Convert PyTorch model to ONNX format using optimum library or alternative methods.
    """
    logger.info(f"Converting model from {model_dir} to ONNX format")
    
    # Create ONNX directory if it doesn't exist
    os.makedirs(onnx_dir, exist_ok=True)
    
    try:
        # Try using optimum if available (cleaner solution)
        from optimum.onnxruntime import ORTModelForCausalLM
        
        logger.info("Using optimum library for ONNX conversion")
        # Load model directly with optimum
        ort_model = ORTModelForCausalLM.from_pretrained(model_dir)
        
        # Save the ONNX model
        ort_model.save_pretrained(onnx_dir)
        
    except ImportError:
        # Fallback to manual conversion with simplified approach
        logger.info("optimum library not found, using manual ONNX conversion")
        
        # Create a simplified export script
        export_script = os.path.join(onnx_dir, "export_script.py")
        with open(export_script, "w") as f:
            f.write("""
import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Get arguments
model_dir = sys.argv[1]
output_dir = sys.argv[2]

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_dir)
tokenizer = AutoTokenizer.from_pretrained(sys.argv[3])

# Setup for export
os.makedirs(output_dir, exist_ok=True)

# Save tokenizer
tokenizer.save_pretrained(output_dir)

# Set up dummy inputs for tracing
batch_size = 1
sequence_length = 32
dummy_input = tokenizer("This is a test", return_tensors="pt", padding="max_length", max_length=sequence_length)

# Export simplified model
class SimpleGPT2ForInference(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, input_ids, attention_mask):
        # Forward pass without past_key_values which causes ONNX export issues
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits

# Create simplified model
simple_model = SimpleGPT2ForInference(model)
simple_model.eval()

# Export to ONNX
torch.onnx.export(
    simple_model,
    (dummy_input["input_ids"], dummy_input["attention_mask"]),
    os.path.join(output_dir, "model.onnx"),
    input_names=["input_ids", "attention_mask"],
    output_names=["logits"],
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
        "logits": {0: "batch_size", 1: "sequence_length"}
    },
    opset_version=13,
    do_constant_folding=True
)

print("ONNX model exported successfully")
            """)
        
        # Run the export script as a separate process
        import subprocess
        tokenizer_path = os.path.join("data", "processed", "tokenizer")
        result = subprocess.run(
            ["python", export_script, model_dir, onnx_dir, tokenizer_path],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            logger.error(f"ONNX export failed: {result.stderr}")
            raise RuntimeError(f"ONNX export failed: {result.stderr}")
        else:
            logger.info(f"ONNX export output: {result.stdout}")
    
    # Copy tokenizer files to ONNX directory for inference
    tokenizer_path = os.path.join("data", "processed", "tokenizer")
    logger.info(f"Copying tokenizer files from {tokenizer_path} to ONNX directory")
    for item in os.listdir(tokenizer_path):
        if item.startswith("tokenizer") or item in ["special_tokens_map.json", "vocab.json", "merges.txt"]:
            src_path = os.path.join(tokenizer_path, item)
            dst_path = os.path.join(onnx_dir, item)
            if os.path.isfile(src_path):
                with open(src_path, 'rb') as src_file, open(dst_path, 'wb') as dst_file:
                    dst_file.write(src_file.read())
    
    onnx_model_path = os.path.join(onnx_dir, "model.onnx")
    logger.info(f"ONNX model conversion completed: {onnx_model_path}")
    
    return onnx_model_path


def verify_onnx_model(onnx_model_path, model_dir):
    """
    Verify the exported ONNX model by comparing outputs with PyTorch model.
    
    Args:
        onnx_model_path (str): Path to the ONNX model
        model_dir (str): Directory containing the PyTorch model
    """
    try:
        import onnxruntime as ort
        
        logger.info("Verifying ONNX model outputs")
        
        # Load PyTorch model and tokenizer
        model = AutoModelForCausalLM.from_pretrained(model_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model.eval()
        
        # Load ONNX model
        ort_session = ort.InferenceSession(onnx_model_path)
        
        # Prepare test input
        test_input = "Genomic variant analysis. Reference sequence: ATGCC. Alternative sequence: ATGAC."
        encoded_input = tokenizer(
            test_input,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=64  # Use a smaller length for verification
        )
        
        # PyTorch forward pass
        with torch.no_grad():
            torch_outputs = model(
                encoded_input["input_ids"],
                attention_mask=encoded_input["attention_mask"]
            )
            torch_logits = torch_outputs.logits.detach().numpy()
        
        # ONNX forward pass
        ort_inputs = {
            "input_ids": encoded_input["input_ids"].numpy(),
            "attention_mask": encoded_input["attention_mask"].numpy()
        }
        ort_outputs = ort_session.run(None, ort_inputs)
        ort_logits = ort_outputs[0]
        
        # Compare outputs (using mean absolute difference)
        import numpy as np
        abs_diff = np.abs(torch_logits - ort_logits).mean()
        
        if abs_diff < 1e-4:
            logger.info(f"ONNX model verification successful! Mean abs diff: {abs_diff}")
            return True
        else:
            logger.warning(f"ONNX model outputs differ from PyTorch model! Mean abs diff: {abs_diff}")
            return False
    
    except ImportError:
        logger.warning("onnxruntime not installed, skipping verification")
        return None
    except Exception as e:
        logger.error(f"Error during ONNX verification: {str(e)}")
        return False


def optimize_onnx_model(onnx_model_path):
    """
    Apply ONNX Runtime optimizations to the model.
    
    Args:
        onnx_model_path (str): Path to the ONNX model
    
    Returns:
        str: Path to the optimized model
    """
    try:
        import onnxruntime as ort
        from onnxruntime.transformers import optimizer
        from onnxruntime.transformers.onnx_model_bert import BertOptimizationOptions
        
        logger.info("Applying ONNX Runtime optimizations")
        
        # Set optimization options
        optimization_options = BertOptimizationOptions("gpt2")
        opt_model_path = onnx_model_path.replace(".onnx", "_optimized.onnx")
        
        # Optimize the model
        optimizer.optimize_model(
            input=onnx_model_path,
            model_type="gpt2",
            num_heads=12,  # Standard for GPT-2
            hidden_size=768,  # Standard for GPT-2
            optimization_options=optimization_options,
            opt_level=1,
            use_gpu=torch.cuda.is_available(),
            only_onnxruntime=True,
            output=opt_model_path
        )
        
        logger.info(f"Optimized ONNX model saved to {opt_model_path}")
        return opt_model_path
    
    except ImportError:
        logger.warning("onnxruntime-transformers not installed, skipping optimization")
        return onnx_model_path
    except Exception as e:
        logger.error(f"Error during ONNX optimization: {str(e)}")
        return onnx_model_path


def main(args):
    """Main function for optimizing the model."""
    logger.info("Starting model optimization process")
    
    # Convert model to ONNX format
    onnx_model_path = convert_to_onnx(
        args.model_dir,
        args.onnx_dir,
        args.max_length
    )
    
    # Verify the ONNX model
    if args.verify:
        success = verify_onnx_model(onnx_model_path, args.model_dir)
        if success is False:
            logger.warning("ONNX verification failed, but continuing with optimization")
    
    # Apply additional optimizations if requested
    if args.optimize:
        try:
            onnx_model_path = optimize_onnx_model(onnx_model_path)
        except Exception as e:
            logger.error(f"Error during additional optimization: {str(e)}")
    
    logger.info("Model optimization complete!")
    logger.info(f"ONNX model saved to: {onnx_model_path}")
    
    return onnx_model_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert the fine-tuned model to ONNX format")
    
    # Model paths
    parser.add_argument("--model_dir", type=str, default=MODEL_DIR, 
                      help="Directory containing the fine-tuned model")
    parser.add_argument("--onnx_dir", type=str, default=ONNX_DIR,
                      help="Directory to save the ONNX model")
    
    # ONNX export options
    parser.add_argument("--max_length", type=int, default=MAX_LENGTH,
                      help="Maximum sequence length for the model")
    parser.add_argument("--verify", action="store_true", default=True,
                      help="Verify the ONNX model outputs against PyTorch")
    parser.add_argument("--optimize", action="store_true", default=True,
                      help="Apply additional ONNX Runtime optimizations")
    
    args = parser.parse_args()
    
    main(args)