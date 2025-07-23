"""
News Summarizer using BART model
Generates concise summaries of news articles (2 paragraphs, ~300 characters)
"""

import torch
from transformers import BartTokenizer, BartForConditionalGeneration
import streamlit as st
from functools import lru_cache
import re

class NewsSummarizer:
    def __init__(self):
        self.model_name = "facebook/bart-large-cnn"
        self.tokenizer = None
        self.model = None
        self._load_model()
    
    @st.cache_resource
    def _load_model(_self):
        """Load BART model and tokenizer with caching"""
        try:
            _self.tokenizer = BartTokenizer.from_pretrained(_self.model_name)
            _self.model = BartForConditionalGeneration.from_pretrained(_self.model_name)
            
            # Move to GPU if available
            if torch.cuda.is_available():
                _self.model = _self.model.cuda()
                
            return _self.tokenizer, _self.model
        except Exception as e:
            st.error(f"Error loading BART model: {str(e)}")
            return None, None
    
    def clean_text(self, text):
        """Clean and preprocess text for summarization"""
        if not text:
            return ""
        
        # Remove extra whitespace and newlines
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove special characters that might interfere with tokenization
        text = re.sub(r'[^\w\s.,!?;:-]', '', text)
        
        return text
    
    def summarize(self, article_text, source_url="", max_length=150):
        """
        Summarize news article to approximately 300 characters (2 paragraphs)
        
        Args:
            article_text (str): Full article text
            source_url (str): URL of the source article
            max_length (int): Maximum tokens for summary
            
        Returns:
            str: Summarized text with source link
        """
        if not self.model or not self.tokenizer:
            return "Summary unavailable - model not loaded"
        
        # Clean the input text
        clean_text = self.clean_text(article_text)
        
        if len(clean_text) < 50:
            return "Article too short for summarization"
        
        try:
            # Tokenize input text
            inputs = self.tokenizer.encode(
                clean_text,
                return_tensors="pt",
                max_length=1024,
                truncation=True
            )
            
            # Move to GPU if available
            if torch.cuda.is_available():
                inputs = inputs.cuda()
            
            # Generate summary
            with torch.no_grad():
                summary_ids = self.model.generate(
                    inputs,
                    max_length=max_length,
                    min_length=30,
                    length_penalty=2.0,
                    num_beams=4,
                    early_stopping=True,
                    do_sample=False
                )
            
            # Decode summary
            summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            
            # Format summary into 2 paragraphs and add source
            summary = self._format_summary(summary, source_url)
            
            return summary
            
        except Exception as e:
            return f"Error generating summary: {str(e)}"
    
    def _format_summary(self, summary, source_url):
        """Format summary into 2 paragraphs and add source link"""
        # Split summary into sentences
        sentences = re.split(r'[.!?]+', summary)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) < 2:
            # If too few sentences, return as is
            formatted_summary = summary
        else:
            # Split into 2 paragraphs
            mid_point = len(sentences) // 2
            paragraph1 = '. '.join(sentences[:mid_point]) + '.'
            paragraph2 = '. '.join(sentences[mid_point:]) + '.'
            formatted_summary = f"{paragraph1}\n\n{paragraph2}"
        
        # Trim to approximately 300 characters if too long
        if len(formatted_summary) > 350:
            formatted_summary = formatted_summary[:300] + "..."
        
        # Add source link if provided
        if source_url:
            formatted_summary += f"\n\n**Source:** [Read full article]({source_url})"
        
        return formatted_summary

# Global instance for reuse
summarizer_instance = None

def get_summarizer():
    """Get or create summarizer instance"""
    global summarizer_instance
    if summarizer_instance is None:
        summarizer_instance = NewsSummarizer()
    return summarizer_instance

@lru_cache(maxsize=100)
def summarize_cached(article_text, source_url=""):
    """Cached version of summarization to avoid re-processing same articles"""
    summarizer = get_summarizer()
    return summarizer.summarize(article_text, source_url)

def summarize_news_batch(articles_data):
    """
    Summarize multiple news articles in batch
    
    Args:
        articles_data (list): List of dicts with 'content' and 'url' keys
        
    Returns:
        list: List of summaries
    """
    summarizer = get_summarizer()
    summaries = []
    
    for article in articles_data:
        content = article.get('content', '')
        url = article.get('url', '')
        summary = summarizer.summarize(content, url)
        summaries.append(summary)
    
    return summaries