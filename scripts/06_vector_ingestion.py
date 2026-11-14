import pandas as pd
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import os

# Paths
DATA_PATH = 'data/processed/useful_support_pairs.csv'
DB_DIR = 'data/chroma_db'

def ingest_to_chroma():
    print("Loading the dataset...")
    df = pd.read_csv(DATA_PATH)
    
    # Drop any edge-case nulls to be safe
    df = df.dropna(subset=['inbound_text', 'outbound_text', 'domain', 'brand'])
    
    print("Initializing CPU-friendly embedding model...")
    # all-MiniLM-L6-v2 is extremely fast and lightweight
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    vector_store = Chroma(
        collection_name="support_triage",
        embedding_function=embeddings,
        persist_directory=DB_DIR
    )
    
    batch_size = 500
    total_rows = len(df)
    
    print(f"Starting ingestion of {total_rows} records in batches of {batch_size}...")
    
    # Process in chunks to prevent RAM overflow
    for i in range(0, total_rows, batch_size):
        batch = df.iloc[i:i+batch_size]
        documents = []
        
        for _, row in batch.iterrows():
            doc = Document(
                page_content=str(row['inbound_text']), # Embed the customer query
                metadata={
                    "outbound_text": str(row['outbound_text']), # The solution
                    "domain": str(row['domain']),
                    "brand": str(row['brand']),
                    "inbound_id": str(row['inbound_tweet_id'])
                }
            )
            documents.append(doc)
            
        vector_store.add_documents(documents)
        print(f"Processed {min(i + batch_size, total_rows)} / {total_rows} records")

    print(f"Ingestion complete. Vector database saved locally in {DB_DIR}.")

if __name__ == "__main__":
    # For testing, you might want to slice df.head(5000) first to see how it performs
    ingest_to_chroma()