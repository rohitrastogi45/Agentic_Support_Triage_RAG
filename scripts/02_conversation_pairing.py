import pandas as pd
import gc

RAW_DATA = 'data/raw/twcs.csv'
MAPPING_FILE = 'data/processed/brand_mapping.csv'
PAIRS_OUT = 'data/processed/support_pairs.csv'

def pair_conversations():
    print("Loading necessary columns to memory...")
    # Load only required columns to fit well within 16GB RAM
    cols = ['tweet_id', 'author_id', 'inbound', 'text', 'in_response_to_tweet_id', 'created_at']
    df = pd.read_csv(RAW_DATA, usecols=cols)
    
    df_inbound = df[df['inbound'] == True].copy()
    df_outbound = df[df['inbound'] == False].copy()
    
    del df # Free memory
    gc.collect()
    
    df_inbound.rename(columns={'tweet_id': 'inbound_tweet_id', 'text': 'inbound_text', 'created_at': 'inbound_created_at'}, inplace=True)
    df_outbound.rename(columns={'tweet_id': 'outbound_tweet_id', 'text': 'outbound_text', 'created_at': 'outbound_created_at'}, inplace=True)
    
    print("Merging conversation pairs...")
    # outbound.in_response_to_tweet_id = inbound.tweet_id
    pairs = pd.merge(
        df_outbound, 
        df_inbound[['inbound_tweet_id', 'inbound_text', 'inbound_created_at', 'author_id']], 
        left_on='in_response_to_tweet_id', 
        right_on='inbound_tweet_id', 
        how='inner',
        suffixes=('_out', '_in')
    )
    
    # Map Domains and Brands
    mapping = pd.read_csv(MAPPING_FILE)
    pairs = pd.merge(pairs, mapping, left_on='author_id_out', right_on='author_id', how='left')
    
    # Cleanup and save
    pairs = pairs[['inbound_tweet_id', 'author_id_in', 'inbound_text', 'inbound_created_at', 
                   'outbound_tweet_id', 'author_id_out', 'brand', 'domain', 'outbound_text', 'outbound_created_at']]
    
    pairs.to_csv(PAIRS_OUT, index=False)
    print(f"Generated {len(pairs)} matched customer-support pairs.")

if __name__ == "__main__":
    pair_conversations()