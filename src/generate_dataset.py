"""
Filename: src/generate_dataset.py
Purpose: Programmatically generate a realistic, synthetic dataset of social media comments
         to be used for training classification, regression, and clustering models in this project.
Explanation: 
- Creates comments labeled as 'Human' (genuine, conversational) or 'IT Cell' (spammy, repetitive, copy-paste).
- Includes numeric features (account age, posting frequency) and target variables (engagement rate, is_it_cell).
- Saves the output as a CSV file in data/raw/social_comments.csv.
"""

import os
import pandas as pd
import numpy as np
import random

def create_synthetic_data(num_samples=1200):
    # Ensure raw directory exists
    os.makedirs("data/raw", exist_ok=True)
    
    # 1. Text samples for Human Comments (diverse, conversational, typo-friendly)
    human_texts = [
        "I absolutely love this new update! The UI looks clean and crisp.",
        "Honestly, I think the movie was slightly overrated, but the acting was top tier.",
        "Can someone explain how to solve the 3rd question in the ML assignment?",
        "Beautiful weather today. Going out for a cup of coffee.",
        "That match last night was insane! What a finish in the last minute.",
        "I don't agree with your opinion, but I appreciate the respectful debate here.",
        "Does anyone know if the local train is running on schedule today?",
        "This recipe looks delicious, definitely trying it out this Sunday!",
        "Struggling with coding errors all day... need a break lol.",
        "The customer service was really helpful and sorted my issue in 5 minutes.",
        "Just finished reading this book. Highly recommend it to everyone.",
        "Wow, that sunset is gorgeous! Thanks for sharing the photo.",
        "I'm not sure if this is correct, but let's see how it goes.",
        "What a great performance by the team. Very happy with the result.",
        "Haha that is so funny, thanks for making my day!"
    ]
    
    # 2. Text samples for IT Cell / Bot / Flooding Comments (repetitive, slogans, aggressive, copy-paste)
    bot_texts = [
        "SUPPORT THIS LEADER FOR BETTER FUTURE! DO NOT TRUST THE OPPOSITION!! #Victory2026",
        "This is the best government in history. Clean sweep in next elections! #PropagandaCampaign",
        "Worst brand ever! Total scam, do not buy their products! SCAMMER!!",
        "CLICK HERE TO EARN $500 DAILY WORKING FROM HOME! 100% REAL LINK IN BIO!!!",
        "Don't listen to this fake news! The truth is hidden by mainstream media!",
        "AMAZING OFFER! GET FREE CRYPTO NOW! CLICK LINK BELOW!!!",
        "This person is a traitor! Stop spreading lies about our great country!",
        "Boycott this movie! It hurts community sentiments! #BoycottTrending",
        "Follow my account for free premium tips and tricks. Guaranteed success!",
        "THE TRUTH HAS BEEN EXPOSED! WATCH THIS VIDEO BEFORE IT GETS DELETED!!!",
        "Unbelievable development! Best leader of all time! Truly visionary leadership!",
        "This is completely fake! Stop spreading negative rumors immediately!",
        "SUPPORT #LEADER2026 NOW! SHARE THIS MESSAGE TO 10 GROUPS NOW!!!",
        "Earn passive income easily. Sign up using code BOT999. Real cash!",
        "This channel is biased and spreading fake propaganda. Ban them!"
    ]
    
    data = []
    
    # Generate random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    for i in range(num_samples):
        # 50% chance of being IT Cell/Bot comment
        is_it_cell = random.choice([0, 1])
        comment_id = f"C{i+1:04d}"
        
        if is_it_cell == 0:
            # Human profile
            comment_text = random.choice(human_texts)
            # Add minor variations (random typos or punctuation) to make it realistic
            if random.random() > 0.7:
                comment_text += " " + random.choice(["😊", "👍", "haha", "...", "!!"])
                
            # Features representing normal user behavior
            account_age_days = int(np.random.normal(loc=800, scale=400))
            account_age_days = max(30, account_age_days) # Min age 30 days
            
            posting_frequency_per_min = round(max(0.05, np.random.normal(loc=0.5, scale=0.3)), 2)
            
            # Text similarity to political templates (should be low for humans)
            similarity_to_trending = round(max(0.0, np.random.normal(loc=0.15, scale=0.1)), 3)
            similarity_to_trending = min(1.0, similarity_to_trending)
            
            # Engagement rate (likes/replies) - humans get normal engagement
            engagement_rate = round(max(1.0, np.random.normal(loc=12.0, scale=5.0)), 2)
            
        else:
            # IT Cell / Bot profile
            comment_text = random.choice(bot_texts)
            if random.random() > 0.6:
                # Bots repeat slogans exactly or add hashtags
                comment_text += " " + random.choice(["#MUSTWATCH", "#ALERT", "#JOINNOW"])
                
            # Features representing spam bot behavior (new accounts, rapid posting)
            account_age_days = int(np.random.exponential(scale=30)) + 1
            account_age_days = min(180, account_age_days) # Mostly new accounts
            
            posting_frequency_per_min = round(max(2.0, np.random.normal(loc=12.0, scale=4.0)), 2)
            
            # High similarity to template text campaigns
            similarity_to_trending = round(max(0.5, np.random.normal(loc=0.8, scale=0.12)), 3)
            similarity_to_trending = min(1.0, similarity_to_trending)
            
            # Engagement rate - spam is either ignored (low) or artificially boosted. Let's make it have
            # a linear relationship with posting frequency for Simple/Multiple Linear Regression modeling.
            # Y (Engagement Rate) = 2.5 * posting_frequency_per_min - 0.05 * account_age_days + noise
            base_engagement = 2.5 * posting_frequency_per_min - 0.05 * account_age_days
            engagement_rate = round(max(0.0, base_engagement + np.random.normal(loc=5.0, scale=3.0)), 2)
            
        data.append({
            "comment_id": comment_id,
            "comment_text": comment_text,
            "account_age_days": account_age_days,
            "posting_frequency_per_min": posting_frequency_per_min,
            "similarity_to_trending": similarity_to_trending,
            "engagement_rate": engagement_rate,
            "is_it_cell": is_it_cell
        })
        
    df = pd.DataFrame(data)
    
    # Save to file
    output_path = "data/raw/social_comments.csv"
    df.to_csv(output_path, index=False)
    print(f"[SUCCESS] Synthetic dataset generated with {num_samples} records and saved to {output_path}")

if __name__ == "__main__":
    create_synthetic_data()
