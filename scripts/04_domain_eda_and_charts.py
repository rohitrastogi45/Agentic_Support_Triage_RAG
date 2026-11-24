import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from transformers import pipeline

PAIRS_FILE = 'data/processed/support_pairs.csv'
REPORTS_DIR = 'reports/domain_eda'
CHARTS_DIR = f'{REPORTS_DIR}/charts'

def generate_eda():
    os.makedirs(CHARTS_DIR, exist_ok=True)
    df = pd.read_csv(PAIRS_FILE)
    
    # Exclude unknowns for clean domain EDA
    df_known = df[df['domain'] != 'Unknown'].copy()
    
    # 1. Customer-support pair distribution by domain
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df_known, x='domain', order=df_known['domain'].value_counts().index)
    plt.title('Customer-Support Pairs by Domain')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'{CHARTS_DIR}/pairs_by_domain.png', dpi=150)
    plt.close()

    # 2. Conversation Length (Proxy using text length)
    df_known['inbound_len'] = df_known['inbound_text'].apply(lambda x: len(str(x)))
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_known, x='domain', y='inbound_len')
    plt.title('Inbound Message Length by Domain')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'{CHARTS_DIR}/inbound_length_by_domain.png', dpi=150)
    plt.close()
    
    # Statistical Exports
    domain_stats = df_known.groupby('domain').agg(
        total_pairs=('inbound_tweet_id', 'count'),
        unique_brands=('brand', 'nunique')
    ).reset_index()
    domain_stats.to_csv(f'{REPORTS_DIR}/domain_distribution.csv', index=False)

    # 3. Issue Category Distribution (Reuse existing CSV if already generated)
    issue_csv = f'{REPORTS_DIR}/issue_category_distribution.csv'
    if os.path.exists(issue_csv):
        print(f"Loading existing {issue_csv} (skipping slow Zero-Shot Classification)...")
        sample_df = pd.read_csv(issue_csv)
    else:
        print("Running Zero-Shot Classification on a representative sample (100 rows per domain)...")
        classifier = pipeline("zero-shot-classification", model="valhalla/distilbart-mnli-12-3")
        categories = ["Account access", "Billing and payments", "Technical support", "Refunds and cancellations", "Delivery and orders", "General inquiries"]
        
        sample_df = df_known.groupby('domain').sample(n=min(100, len(df_known)), random_state=42)
        
        def get_category(text):
            try:
                result = classifier(text, candidate_labels=categories)
                return result['labels'][0]
            except:
                return "Other"
                
        sample_df['pseudo_issue_category'] = sample_df['inbound_text'].apply(get_category)
        sample_df.to_csv(issue_csv, index=False)
    
    # Horizontal bar chart: tight_layout() fixes the clipped y-axis labels
    plt.figure(figsize=(12, 6))
    sns.countplot(data=sample_df, y='pseudo_issue_category', hue='domain')
    plt.title('Sample Issue Category Distribution by Domain')
    plt.tight_layout()
    plt.savefig(f'{CHARTS_DIR}/issue_category_sample.png', dpi=150)
    plt.close()
    
    print("EDA Generation Complete. Charts saved to reports/domain_eda/charts/")

if __name__ == "__main__":
    generate_eda()