#!/usr/bin/env python
"""Streamlit app for the frontend interface."""

import os
import json
import time
import logging
import requests
import streamlit as st

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_API_URL = "http://localhost:8000"  # Default URL for local testing
MAX_NEW_TOKENS = 50
DEFAULT_GENOMIC_TEXT = """
Genomic variant analysis. Reference sequence: ATGCTGCAATCGATCGTAGCTGCTAGCTAGCTAGCTAGCTAGCTAGCTATGCATGCATGCATGCA. 
Alternative sequence: ATGCTGCAATCGATCGTAGCTGCTAGCTAGCTAGCTTGCTAGCTAGCTATGCATGCATGCATGCA. 
Chromosome: 7. Position: 140453136. 
This variant affects a promoter region that is associated with gene expression processes.
"""

def create_sidebar():
    """Create the sidebar with app information and settings."""
    with st.sidebar:
        st.title("About")
        st.write(
            "This application uses a fine-tuned language model to summarize "
            "genomic research text. The model is deployed as a REST API."
        )
        
        st.header("Settings")
        api_url = st.text_input(
            "API URL",
            value=os.environ.get("API_URL", DEFAULT_API_URL),
            help="URL of the deployed API. Default is localhost for local testing."
        )
        
        max_tokens = st.slider(
            "Maximum Summary Length",
            min_value=10,
            max_value=100,
            value=MAX_NEW_TOKENS,
            step=10,
            help="Maximum number of tokens to generate for the summary."
        )
        
        return api_url, max_tokens

def summarize_text(text, api_url, max_new_tokens):
    """
    Send the text to the API for summarization.
    Args:
        text (str): Input text to summarize
        api_url (str): API URL
        max_new_tokens (int): Maximum number of new tokens to generate
    Returns:
        dict: API response containing the summary
    """
    if not text:
        return {"error": "No text provided"}
    
    if not api_url:
        return {"error": "No API URL provided"}
    
    try:
        # Prepare the request
        endpoint = f"{api_url.rstrip('/')}/summarize"
        payload = {
            "text": text,
            "max_new_tokens": max_new_tokens
        }
        headers = {"Content-Type": "application/json"}
        
        # Send the request to the API
        start_time = time.time()
        response = requests.post(endpoint, json=payload, headers=headers)
        response_time = time.time() - start_time
        
        # Check if the request was successful
        if response.status_code == 200:
            result = response.json()
            result["response_time"] = response_time
            return result
        else:
            return {
                "error": f"API request failed with status code {response.status_code}",
                "details": response.text,
                "response_time": response_time
            }
    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}

def generate_fallback_summary(text):
    """
    Generate a simple summary if API is not available.
    This is just for demonstration purposes.
    """
    import random
    
    fallback_summaries = [
        "This genomic variant may be pathogenic.",
        "The genomic sequence shows significant expression patterns.",
        "This variant affects a regulatory element in the genome.",
        "Analysis indicates this sequence has important functional properties."
    ]
    
    # Return a random fallback summary
    return {
        "summary": random.choice(fallback_summaries),
        "model_type": "Fallback",
        "response_time": 0.01
    }

def main():
    """Main function for the Streamlit app."""
    st.set_page_config(
        page_title="Genomics Text Summarizer",
        page_icon="🧬",
        layout="wide"
    )
    
    # Create sidebar and get settings
    api_url, max_tokens = create_sidebar()
    
    # Main content
    st.title("Genomics Text Summarizer")
    st.write(
        "Enter genomic research text below to generate a concise summary. "
        "The AI model is trained specifically on genomic and scientific text."
    )
    
    # Text input area
    text_input = st.text_area(
        "Genomic Text",
        value=DEFAULT_GENOMIC_TEXT,
        height=200,
        help="Paste genomic research text here for summarization."
    )
    
    # Submit button and API check options
    col1, col2 = st.columns([1, 1])
    with col1:
        submit_button = st.button("Generate Summary", type="primary")
    
    with col2:
        use_api = st.checkbox("Use API (uncheck for offline demo)", value=True)
    
    # Check API connection
    if st.button("Check API Connection", type="secondary"):
        if not use_api:
            st.info("API check skipped - running in offline demo mode")
        else:
            try:
                # Use the health endpoint to check if the API is running
                endpoint = f"{api_url.rstrip('/')}/health"
                response = requests.get(endpoint)
                
                if response.status_code == 200:
                    st.success(f"✅ API connection successful: {response.json()}")
                else:
                    st.error(f"❌ API connection failed: Status code {response.status_code}")
            except Exception as e:
                st.error(f"❌ API connection failed: {str(e)}")
    
    # Process the text when the submit button is clicked
    if submit_button:
        if not text_input.strip():
            st.warning("Please enter some text to summarize.")
        else:
            with st.spinner("Generating summary..."):
                if use_api:
                    result = summarize_text(text_input, api_url, max_tokens)
                else:
                    # Use fallback for demonstration if API is not available
                    result = generate_fallback_summary(text_input)
            
            if "error" in result:
                st.error(f"Error: {result['error']}")
                if "details" in result:
                    with st.expander("Error Details"):
                        st.code(result["details"])
                # Show fallback summary if API fails
                st.warning("Showing demo summary instead:")
                result = generate_fallback_summary(text_input)
            
            # Display the summary
            st.subheader("Generated Summary")
            st.success(result["summary"])
            
            # Display additional information
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"Model Type: {result.get('model_type', 'Unknown')}")
            with col2:
                st.info(f"Processing Time: {result.get('response_time', 0):.2f} seconds")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "This application is part of the Genomics Text Summarization project. "
        "The model is fine-tuned on genomics-related text data to produce concise summaries."
    )

if __name__ == "__main__":
    main()