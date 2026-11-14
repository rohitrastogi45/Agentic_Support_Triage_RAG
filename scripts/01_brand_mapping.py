import pandas as pd
import os

RAW_DATA = 'data/raw/twcs.csv'
TEMPLATE_OUT = 'data/processed/brand_mapping_template.csv'
MAPPING_OUT = 'data/processed/brand_mapping.csv'

def generate_brand_mapping():
    os.makedirs('data/processed', exist_ok=True)
    
    print("Extracting unique support accounts in chunks...")
    unique_outbound_authors = set()
    
    # Chunking to respect 16GB RAM limit
    chunk_size = 100000
    for chunk in pd.read_csv(RAW_DATA, usecols=['author_id', 'inbound'], chunksize=chunk_size):
        outbound = chunk[chunk['inbound'] == False]
        unique_outbound_authors.update(outbound['author_id'].unique())
        
    df_mapping = pd.DataFrame({'author_id': list(unique_outbound_authors)})
    
    # Pre-populate known examples for the prototype
    known_mappings = {
        'AppleSupport': ('Apple', 'Technology'),
        'AmazonHelp': ('Amazon', 'E-commerce'),
        'Delta': ('Delta Airlines', 'Airlines'),
        'SouthwestAir': ('Southwest', 'Airlines'),
        'SpotifyCares': ('Spotify', 'Technology')
    }
    
    df_mapping['brand'] = df_mapping['author_id'].apply(lambda x: known_mappings.get(x, ('Unknown', 'Unknown'))[0])
    df_mapping['domain'] = df_mapping['author_id'].apply(lambda x: known_mappings.get(x, ('Unknown', 'Unknown'))[1])
    
    df_mapping.to_csv(TEMPLATE_OUT, index=False)
    df_mapping.to_csv(MAPPING_OUT, index=False)
    print(f"Mapping saved to {MAPPING_OUT}. Update 'Unknown' values manually if needed.")

if __name__ == "__main__":
    generate_brand_mapping()