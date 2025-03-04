#!/usr/bin/env python
"""Training script for fine-tuning the language model."""

import shutil
import os
import pickle
import logging
import argparse
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AdamW,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants - IMPROVED FOR 1-HOUR TRAINING
MODEL_NAME = "gpt2"  # Using the full GPT-2 model (still manageable size)
MAX_LENGTH = 384  # Increased from 256 for more context
BATCH_SIZE = 4  # Adjust based on your machine's memory
EPOCHS = 3  # Multiple epochs for better learning
LEARNING_RATE = 3e-5  # Slightly reduced for better stability
WARMUP_RATIO = 0.1  # Proportion of training steps for warmup
GRADIENT_ACCUMULATION_STEPS = 2  # Simulate larger batch size
PROCESSED_DIR = os.path.join("data", "processed")
MODEL_DIR = os.path.join("models", "checkpoints")
MAX_SAMPLES = 1000  # Increased sample size for better learning

class GenomicsTextDataset(Dataset):
    """Dataset class for the tokenized genomics text data."""
    
    def __init__(self, tokenized_data, max_samples=None):
        """
        Initialize the dataset.
        Args:
            tokenized_data (dict): Tokenized data from preprocessing
            max_samples (int): Maximum number of samples to include
        """
        self.input_ids = tokenized_data["input_ids"]
        self.attention_mask = tokenized_data["attention_mask"]
        
        # Limit the number of samples if specified
        if max_samples and max_samples < len(self.input_ids):
            logger.info(f"Using {max_samples} samples for training")
            self.input_ids = self.input_ids[:max_samples]
            self.attention_mask = self.attention_mask[:max_samples]
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.input_ids[idx].clone()  # For causal language modeling
        }

def load_data(max_samples=MAX_SAMPLES):
    """
    Load the preprocessed tokenized data.
    Returns:
        tuple: (train_dataset, val_dataset)
    """
    logger.info("Loading preprocessed data...")
    
    # Load tokenized training data
    with open(os.path.join(PROCESSED_DIR, "train_tokenized.pkl"), "rb") as f:
        train_tokenized = pickle.load(f)
    
    # Load tokenized validation data
    with open(os.path.join(PROCESSED_DIR, "val_tokenized.pkl"), "rb") as f:
        val_tokenized = pickle.load(f)
    
    # Create datasets with appropriate sample size
    train_dataset = GenomicsTextDataset(train_tokenized, max_samples)
    val_dataset = GenomicsTextDataset(val_tokenized, max_samples // 5)  # 20% of train size
    
    logger.info(f"Loaded {len(train_dataset)} training and {len(val_dataset)} validation samples")
    
    return train_dataset, val_dataset

def load_model_and_tokenizer():
    """
    Load the pre-trained model and tokenizer.
    Returns:
        tuple: (model, tokenizer)
    """
    logger.info(f"Loading pretrained {MODEL_NAME} model...")
    
    # Load pretrained model
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(PROCESSED_DIR, "tokenizer"),
        model_max_length=MAX_LENGTH
    )
    
    # Set padding token if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    
    logger.info(f"Model loaded: {model.config.model_type}")
    
    return model, tokenizer

def train(model, train_dataloader, val_dataloader, device, args):
    """
    Fine-tune the model.
    Args:
        model: The pre-trained model
        train_dataloader: Training data loader
        val_dataloader: Validation data loader
        device: Device to train on
        args: Command-line arguments
    """
    logger.info(f"Training on {device}")
    
    # Move model to device
    model.to(device)
    
    # Set up optimizer with weight decay
    # Apply weight decay to all parameters except bias and layer norm
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)
    
    # Calculate total training steps
    total_steps = len(train_dataloader) * args.epochs // args.gradient_accumulation_steps
    warmup_steps = int(total_steps * args.warmup_ratio)
    
    # Use cosine schedule with warmup
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps, 
        num_training_steps=total_steps
    )
    
    # Track best model
    best_val_loss = float('inf')
    
    # Training loop
    for epoch in range(args.epochs):
        logger.info(f"Starting epoch {epoch + 1}/{args.epochs}")
        
        # Training
        model.train()
        train_loss = 0
        train_steps = 0
        optimizer.zero_grad()  # Reset gradients at the start of each epoch
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")
        for step, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss / args.gradient_accumulation_steps  # Normalize loss
            
            # Backward pass
            loss.backward()
            
            # Update weights every gradient_accumulation_steps
            if (step + 1) % args.gradient_accumulation_steps == 0:
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            # Update metrics
            train_loss += loss.item() * args.gradient_accumulation_steps  # De-normalize for logging
            train_steps += 1
            
            # Update progress bar
            progress_bar.set_postfix({"loss": loss.item() * args.gradient_accumulation_steps})
        
        avg_train_loss = train_loss / train_steps
        logger.info(f"Epoch {epoch + 1} - Average training loss: {avg_train_loss:.4f}")
        
        # Validation
        model.eval()
        val_loss = 0
        val_steps = 0
        
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="Validating"):
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                
                val_loss += loss.item()
                val_steps += 1
        
        avg_val_loss = val_loss / val_steps
        logger.info(f"Epoch {epoch + 1} - Average validation loss: {avg_val_loss:.4f}")
        
        # Save checkpoint for the epoch
        model_path = os.path.join(MODEL_DIR, f"genomics_summarizer_epoch_{epoch + 1}")
        model.save_pretrained(model_path)
        logger.info(f"Saved epoch checkpoint to {model_path}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_path = os.path.join(MODEL_DIR, "genomics_summarizer_best")
            model.save_pretrained(model_path)
            logger.info(f"New best model with validation loss: {best_val_loss:.4f}")
    
    logger.info("Training complete!")
    return model

def main(args):
    """Main training function."""
    logger.info("Starting training process (improved 1-hour version)...")
    
    # Set up the device
    if args.force_cpu:
        device = torch.device("cpu")
        logger.info("Forcing CPU usage as requested")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Using CUDA GPU")
    elif torch.backends.mps.is_available() and not args.force_cpu:
        device = torch.device("mps")
        logger.info("Using MPS (Apple Silicon GPU)")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")
    
    # Create model directory
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Load data with increased sample size
    train_dataset, val_dataset = load_data(max_samples=args.max_samples)
    
    # Create data loaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer()
    
    # Train the model with improved settings
    model = train(model, train_dataloader, val_dataloader, device, args)
    
    # Save the final model
    final_model_path = os.path.join(MODEL_DIR, "genomics_summarizer_final")
    model.save_pretrained(final_model_path)
    logger.info(f"Saved final model to {final_model_path}")
    
    # Copy the best model to final if requested
    if args.use_best_model:
        best_model_path = os.path.join(MODEL_DIR, "genomics_summarizer_best")
        if os.path.exists(best_model_path):

            if os.path.exists(final_model_path):
                shutil.rmtree(final_model_path)  # Remove existing directory first

            shutil.copytree(best_model_path, final_model_path)  # Now copy safely
            logger.info(f"Copied best model to {final_model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a language model for genomics text summarization")
    parser.add_argument("--model", type=str, default=MODEL_NAME, help="Model name to use")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Learning rate")
    parser.add_argument("--max_samples", type=int, default=MAX_SAMPLES, help="Maximum number of samples to use")
    parser.add_argument("--force_cpu", action="store_true", help="Force CPU usage even if GPU is available")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=GRADIENT_ACCUMULATION_STEPS, 
                        help="Number of updates steps to accumulate before backward pass")
    parser.add_argument("--warmup_ratio", type=float, default=WARMUP_RATIO, 
                        help="Proportion of training steps for learning rate warmup")
    parser.add_argument("--num_workers", type=int, default=0, 
                        help="Number of workers for data loading")
    parser.add_argument("--use_best_model", action="store_true", default=True,
                        help="Use the best model based on validation loss as the final model")
    
    args = parser.parse_args()
    
    main(args)