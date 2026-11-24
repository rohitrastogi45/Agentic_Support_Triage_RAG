import pandas as pd
import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

total_pairs = len(pd.read_csv("data/processed/support_pairs.csv"))
useful_pairs = len(pd.read_csv("data/processed/useful_support_pairs.csv"))
review_path = "data/processed/review_support_pairs.csv"
review_pairs = len(pd.read_csv(review_path)) if os.path.exists(review_path) else 0
dropped_pairs = total_pairs - useful_pairs - review_pairs

# Verify actual ChromaDB count
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
db = Chroma(collection_name="support_triage", embedding_function=embeddings, persist_directory="data/chroma_db")
embedded_count = db._collection.count()

funnel_df = pd.DataFrame([
    {"Stage": "1. Master Paired Corpus (support_pairs.csv)", "Count": total_pairs, "Percentage": "100.00%"},
    {"Stage": "2. Dropped (DM Us / Generic Apologies)", "Count": dropped_pairs, "Percentage": f"{dropped_pairs/total_pairs*100:.2f}%"},
    {"Stage": "3. QC Review Queue (review_support_pairs.csv)", "Count": review_pairs, "Percentage": f"{review_pairs/total_pairs*100:.2f}%"},
    {"Stage": "4. Verified Solutions (useful_support_pairs.csv)", "Count": useful_pairs, "Percentage": f"{useful_pairs/total_pairs*100:.2f}%"},
    {"Stage": "5. Embedded in ChromaDB (support_triage)", "Count": embedded_count, "Percentage": f"{embedded_count/total_pairs*100:.2f}%"}
])

print(funnel_df.to_string(index=False))
funnel_df.to_csv("reports/domain_eda/quality_filter_funnel.csv", index=False)