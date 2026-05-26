"""
Filename: src/data_preprocessing.py
Purpose: Perform text cleaning and numeric feature preprocessing.
Explanation:
- Cleans raw text: converts to lowercase, removes special characters, and removes extra spaces.
- Preprocesses numeric features: scales features like account age and posting frequency.
- Explains every step in a beginner-friendly manner.
"""

import re
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def clean_text(text):
    """
    Cleans raw text comments for NLP classification.
    Steps:
    1. Converts text to lowercase to ensure consistency (e.g., 'GET' and 'get' are treated the same).
    2. Removes HTML patterns or URLs if any.
    3. Removes special characters/punctuation except spaces.
    4. Trims consecutive whitespaces.
    """
    # Check if text is valid string
    if not isinstance(text, str):
        return ""
    
    # Step 1: Lowercase
    text = text.lower()
    
    # Step 2: Remove URLs/links
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Step 3: Remove special characters and numbers (keep letters and spaces)
    text = re.sub(r'[^a-zA-Z\s#]', '', text)
    
    # Step 4: Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def preprocess_dataframe(df):
    """
    Cleans text and scales numerical attributes of the comments dataframe.
    """
    # Create a copy of the dataframe to avoid setting with copy warning
    processed_df = df.copy()
    
    # 1. Clean the text column
    processed_df['cleaned_text'] = processed_df['comment_text'].apply(clean_text)
    
    # 2. Scale numeric columns
    # We will use MinMaxScaler for age and Standard Scaler for frequency to show both methods!
    # MinMaxScaler maps data strictly between 0 and 1: (x - min) / (max - min)
    # StandardScaler maps data to mean=0 and variance=1: (x - mean) / std
    
    age_scaler = MinMaxScaler()
    freq_scaler = StandardScaler()
    
    # Reshaping input to 2D array as required by sklearn scalers
    processed_df['scaled_age'] = age_scaler.fit_transform(processed_df[['account_age_days']])
    processed_df['scaled_frequency'] = freq_scaler.fit_transform(processed_df[['posting_frequency_per_min']])
    processed_df['scaled_similarity'] = age_scaler.fit_transform(processed_df[['similarity_to_trending']])
    
    return processed_df, age_scaler, freq_scaler

if __name__ == "__main__":
    # Test block
    print("Testing Preprocessing functions...")
    sample_text = "CHECK out this website! https://example.com/promo !! Earn $1000 now!!"
    print("Original Text:", sample_text)
    print("Cleaned Text:", clean_text(sample_text))
