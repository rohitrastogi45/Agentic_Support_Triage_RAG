import pandas as pd
import numpy as np
import re
import os

PAIRS_FILE = 'data/processed/support_pairs.csv'
USEFUL_OUT = 'data/processed/useful_support_pairs.csv'
REVIEW_OUT = 'data/processed/review_support_pairs.csv'
PARTITIONS_DIR = 'data/processed/partitions'

def classify_reply(text):
    text = str(text).lower()
    
    generic_patterns = [r'dm us', r'direct message', r'send us a dm', r'private message', r'reach out to us on dm']
    ack_patterns = [r'^hi.*we are looking into this', r'^sorry for the inconvenience', r'^thanks for reaching out$']
    
    if any(re.search(p, text) for p in generic_patterns):
        return 'Generic Deflection'
    elif any(re.search(p, text) for p in ack_patterns):
        return 'Acknowledgement'
    elif 'http' in text and len(text.split()) > 10:
        return 'Actionable Response'
    else:
        return 'Uncertain'

def process_quality_and_partition():
    print("Loading pairs for quality classification...")
    df = pd.read_csv(PAIRS_FILE)
    
    df['reply_quality'] = df['outbound_text'].apply(classify_reply)
    
    useful_df = df[df['reply_quality'] == 'Actionable Response']
    review_df = df[df['reply_quality'] == 'Uncertain']
    
    # These two lines generate your missing files
    useful_df.to_csv(USEFUL_OUT, index=False)
    review_df.to_csv(REVIEW_OUT, index=False)
    print(f"Saved {len(useful_df)} useful pairs and {len(review_df)} review pairs.")
    
    print("Partitioning useful data by domain and brand...")
    for domain in useful_df['domain'].dropna().unique():
        if domain == 'Unknown': continue
        
        # Sanitize domain name for folder creation
        safe_domain = re.sub(r'[\\/*?:"<>|]', "", domain.lower().replace(' ', '_'))
        domain_dir = os.path.join(PARTITIONS_DIR, safe_domain)
        os.makedirs(domain_dir, exist_ok=True)
        
        domain_df = useful_df[useful_df['domain'] == domain]
        for brand in domain_df['brand'].unique():
            # Sanitize brand name for file creation
            safe_brand = re.sub(r'[\\/*?:"<>|]', "", brand.lower().replace(' ', '_'))
            brand_file = os.path.join(domain_dir, f"{safe_brand}.csv")
            domain_df[domain_df['brand'] == brand].to_csv(brand_file, index=False)

if __name__ == "__main__":
    process_quality_and_partition()