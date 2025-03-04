#!/usr/bin/env python
"""Inference script to generate summaries from genomic text."""

import os
import logging
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
MODEL_DIR = os.path.join("models", "checkpoints", "genomics_summarizer_final")
TOKENIZER_DIR = os.path.join("data", "processed", "tokenizer")
MAX_LENGTH = 256
MAX_NEW_TOKENS = 50  # Reduced for faster inference

class GenomicsSummarizer:
    """Class for generating summaries from genomic text."""
    
    def __init__(self, model_path=None, tokenizer_path=None):
        """
        Initialize the summarizer.
        Args:
            model_path (str): Path to the fine-tuned model
            tokenizer_path (str): Path to the tokenizer
        """
        if model_path is None:
            model_path = MODEL_DIR
        
        if tokenizer_path is None:
            tokenizer_path = TOKENIZER_DIR
        
        # Load tokenizer
        logger.info(f"Loading tokenizer from {tokenizer_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        logger.info(f"Loading PyTorch model from {model_path}")
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.model.eval()
    
    def _extract_summary(self, text):
        """
        Extract the summary from the generated text.
        Args:
            text (str): Generated text
        Returns:
            str: Extracted summary
        """
        # Extract the summary part after "summary:"
        if "summary:" in text.lower():
            return text.split("summary:", 1)[1].strip()
        return text.strip()
    
    def summarize(self, text, max_new_tokens=MAX_NEW_TOKENS):
        """
        Generate a summary from the input text.
        Args:
            text (str): Input text to summarize
            max_new_tokens (int): Maximum number of new tokens to generate
        Returns:
            str: Generated summary
        """
        # Prepare input
        prompt = f"summarize: {text} summary:"
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH - max_new_tokens
        )
        
        # Generate summary
        with torch.no_grad():
            outputs = self.model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_new_tokens,
                num_beams=2,  # Reduced from 4 for faster inference
                no_repeat_ngram_size=2,
                early_stopping=True
            )
        
        # Decode the output
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract the summary part
        summary = self._extract_summary(generated_text)
        
        return summary


def main(args):
    """Main inference function."""
    logger.info("Starting inference...")
    
    # Initialize summarizer
    summarizer = GenomicsSummarizer(
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path
    )
    
    # Get input text from file or argument
    if args.input_file:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            text = f.read()
    else:
        text = args.text
    
    # Generate summary
    logger.info("Generating summary...")
    summary = summarizer.summarize(text, max_new_tokens=args.max_new_tokens)
    
    # Print the summary
    print("\nGenerated Summary:")
    print("-" * 40)
    print(summary)
    print("-" * 40)
    
    # Save the summary to file if requested
    if args.output_file:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            f.write(summary)
        logger.info(f"Summary saved to {args.output_file}")
    
    logger.info("Inference complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate summaries from genomic text")
    parser.add_argument("--model_path", type=str, default=MODEL_DIR, help="Path to the fine-tuned model")
    parser.add_argument("--tokenizer_path", type=str, default=TOKENIZER_DIR, help="Path to the tokenizer")
    parser.add_argument("--text", type=str, default=None, help="Input text to summarize")
    parser.add_argument("--input_file", type=str, default=None, help="Path to input text file")
    parser.add_argument("--output_file", type=str, default=None, help="Path to save the summary")
    parser.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS, help="Maximum number of new tokens to generate")
    
    args = parser.parse_args()
    
    # Ensure text is provided
    if args.text is None and args.input_file is None:
        parser.error("Either --text or --input_file must be provided")
    
    main(args)