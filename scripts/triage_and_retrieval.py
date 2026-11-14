import os
from transformers import pipeline
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Define your local database directory
DB_DIR = "chroma_db"

class TriageAgent:
    def __init__(self):
        print("Loading Triage Classifier...")
        # Using the zero-shot classifier
        self.classifier = pipeline(
            "zero-shot-classification", 
            model="valhalla/distilbart-mnli-12-3",
            device=-1 # Forces CPU
        )
        
        # 1. Domain Mapping (Fixes the Cross-Domain Confusion bug)
        self.domain_mapping = {
            "flight booking, travel, and airlines": "Airlines",
            "computers, software, and technology": "Technology",
            "online shopping, shipping, and retail": "E-commerce"
        }
        self.domains = list(self.domain_mapping.keys())
        
        # 2. Urgency Mapping (Fixes the 'Crying Wolf' false emergency bug)
        self.urgency_mapping = {
            "technical system outage, website down, payment gateway failure, or server crash": "system outage or critical emergency",
            "wrong item received, cosmetic defects, returns, standard troubleshooting, and product support": "standard troubleshooting",
            "general information, store hours, and policy questions": "general information"
        }
        self.urgency_levels = list(self.urgency_mapping.keys())
        
        print("Connecting to ChromaDB...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        self.vector_store = Chroma(
            collection_name="support_triage",
            embedding_function=self.embeddings,
            persist_directory=DB_DIR
        )

    def route_query(self, query):
        print("\n--- Running Triage ---")
        # Identify Domain
        domain_result = self.classifier(query, candidate_labels=self.domains)
        best_match_label = domain_result['labels'][0]
        predicted_domain = self.domain_mapping[best_match_label]
        
        # Identify Urgency using the updated strict labels
        urgency_result = self.classifier(query, candidate_labels=self.urgency_levels)
        best_urgency_label = urgency_result['labels'][0]
        
        # Map it back to the standard urgency category
        final_urgency = self.urgency_mapping[best_urgency_label]
        is_urgent = final_urgency == "system outage or critical emergency"
        
        print(f"Detected Domain: **{predicted_domain}**")
        print(f"Urgency Status: {'High Priority - Human Review Recommended' if is_urgent else 'Standard Processing'}")
        
        return predicted_domain, is_urgent

    def retrieve_solutions(self, query, domain):
        print(f"\n--- Retrieving past solutions for {domain} ---")
        
        # Search using the exact mapped domain
        retriever = self.vector_store.as_retriever(
            search_kwargs={
                "k": 3, 
                "filter": {"domain": domain} 
            }
        )
        
        results = retriever.invoke(query)
        
        if not results:
            print("No relevant historical solutions found.")
            return []
            
        for idx, doc in enumerate(results, 1):
            print(f"\nMatch {idx} (Brand: {doc.metadata.get('brand')}):")
            print(f"Historical Customer Query: {doc.page_content}")
            
            # Truncating the printed solution slightly for cleaner terminal logs
            full_solution = doc.metadata.get('outbound_text', '')
            display_solution = full_solution if len(full_solution) < 150 else full_solution[:147] + "..."
            print(f"Agent Solution: {display_solution}")
            
        return results