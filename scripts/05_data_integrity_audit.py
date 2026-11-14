import pandas as pd
import os

PAIRS_FILE = 'data/processed/support_pairs.csv'
REPORTS_DIR = 'reports/domain_eda'

def run_integrity_audit():
    print("Loading master pairs for integrity audit...")
    df = pd.read_csv(PAIRS_FILE)
    
    # 1. Missing Value Analysis
    missing_df = df.isnull().sum().reset_index()
    missing_df.columns = ['Column', 'Missing_Count']
    missing_df['Missing_Percentage'] = (missing_df['Missing_Count'] / len(df)) * 100
    missing_df.to_csv(f'{REPORTS_DIR}/missing_value_analysis.csv', index=False)
    
    # 2. Duplicate Analysis (Checking for duplicate conversation threads)
    duplicate_count = df.duplicated(subset=['inbound_tweet_id', 'outbound_tweet_id']).sum()
    dup_df = pd.DataFrame([{
        'Total_Pairs': len(df), 
        'Exact_Duplicates': duplicate_count,
        'Duplicate_Percentage': (duplicate_count / len(df)) * 100
    }])
    dup_df.to_csv(f'{REPORTS_DIR}/duplicate_analysis.csv', index=False)
    
    print(f"Audit complete! Reports saved to {REPORTS_DIR}.")

if __name__ == "__main__":
    run_integrity_audit()