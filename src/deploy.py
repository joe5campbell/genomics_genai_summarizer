#!/usr/bin/env python
"""FastAPI deployment script."""

import os
import logging
from typing import Dict, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# Import our inference module
from src.inference import GenomicsSummarizer

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
MAX_NEW_TOKENS = 50

# Create the FastAPI app
app = FastAPI(
    title="Genomics Text Summarization API",
    description="API for summarizing genomic research text",
    version="1.0.0"
)

# Define request and response models
class SummarizationRequest(BaseModel):
    text: str
    max_new_tokens: Optional[int] = MAX_NEW_TOKENS

class SummarizationResponse(BaseModel):
    summary: str
    model_type: str

# Initialize the summarizer
summarizer = None

@app.on_event("startup")
async def startup_event():
    """Initialize the model on startup."""
    global summarizer
    
    try:
        logger.info(f"Initializing summarizer with PyTorch model from {MODEL_DIR}")
        summarizer = GenomicsSummarizer(
            model_path=MODEL_DIR,
            tokenizer_path=TOKENIZER_DIR
        )
    except Exception as e:
        logger.error(f"Error initializing summarizer: {str(e)}")
        raise

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Genomics Text Summarization API",
        "status": "online",
        "model_type": "PyTorch"
    }

@app.get("/health")
async def health():
    """Health check endpoint."""
    if summarizer is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    return {"status": "healthy"}

@app.post("/summarize", response_model=SummarizationResponse)
async def summarize(request: SummarizationRequest):
    """
    Summarize genomic text.
    Args:
        request: SummarizationRequest containing text and options
    Returns:
        SummarizationResponse with the generated summary
    """
    if summarizer is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    
    try:
        # Generate summary
        summary = summarizer.summarize(
            request.text,
            max_new_tokens=request.max_new_tokens
        )
        
        # Return the response
        return SummarizationResponse(
            summary=summary,
            model_type="PyTorch"
        )
    except Exception as e:
        logger.error(f"Error generating summary: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models")
async def models():
    """Get information about available models."""
    models_info = {
        "pytorch_model": {
            "path": MODEL_DIR,
            "available": os.path.exists(MODEL_DIR)
        },
        "current_model": "PyTorch"
    }
    return models_info

def main():
    """Run the API server."""
    uvicorn.run("src.deploy:app", host="0.0.0.0", port=8000, reload=False)

if __name__ == "__main__":
    main()