import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import torch

def clean_currency_and_numbers(df, columns):
    """Utility function to strip currency symbols, commas, and handle NaNs."""
    for col in columns:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(r'[₹,]', '', regex=True)
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
    return df

def clean_string_columns(df, columns):
    """Utility function to trim whitespace and standardize casing."""
    for col in columns:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.upper()
    return df

def load_and_merge_mplads_datasets():
    print("[*] Loading all 6 MPLADS Datasets...")

    df1_allocated = pd.read_csv("assests/Allocated Limit for Honble MPs.csv")
    df2_calamity = pd.read_csv("assests/Amount consented for Calamity.csv")
    df3_recommended = pd.read_csv("assests/Works Recommended.csv")
    df4_sanctioned = pd.read_csv("assests/Works Sanctioned.csv")
    df5_completed = pd.read_csv("assests/Works Completed.csv")
    df6_expenditure = pd.read_csv("assests/Expenditure on Completed and On-going Works as on Date.csv")

    #simple cleaning of column names to remove leading/trailing spaces and standardize casing
    df1_allocated.columns = df1_allocated.columns.str.strip().str.title()
    df1_allocated = df1_allocated.drop(columns=['Sr. No.'], errors='ignore')
    df2_calamity.columns = df2_calamity.columns.str.strip().str.title()
    df2_calamity = df2_calamity.drop(columns=['Sr. No.'], errors='ignore')
    df3_recommended.columns = df3_recommended.columns.str.strip().str.title()
    df3_recommended = df3_recommended.drop(columns=['Sr. No.'], errors='ignore')
    df4_sanctioned.columns = df4_sanctioned.columns.str.strip().str.title()
    df4_sanctioned = df4_sanctioned.drop(columns=['Sr. No.'], errors='ignore')
    df5_completed.columns = df5_completed.columns.str.strip().str.title()
    df5_completed = df5_completed.drop(columns=['Sr. No.'], errors='ignore')
    df6_expenditure.columns = df6_expenditure.columns.str.strip().str.title()
    df6_expenditure = df6_expenditure.drop(columns=['Sr. No.'], errors='ignore')
    df1_allocated = df1_allocated.rename(columns={"Hon'Ble Members Of Parliaments": "Hon'Ble Members Of Parliament"})
    df6_expenditure['Work'] = df6_expenditure['Work Id'] + '-' + df6_expenditure['Work']

    print("[*] Standardizing and Cleaning Individual Datasets...")

    # =========================================================================
    # 2. CLEAN & AGGREGATE FINANCIALS (DATASETS 1 & 2)
    # Key: ['mp_id', 'financial_year']
    # =========================================================================
    df1_allocated = clean_currency_and_numbers(df1_allocated, ['Allocated Amount ( ₹ )'])
    df1_allocated = clean_string_columns(df1_allocated, ["Hon'Ble Members Of Parliament", 'Constituency'])

    df2_calamity = clean_currency_and_numbers(df2_calamity, ['Consent Amount ( ₹ )'])
    df2_calamity = clean_string_columns(df2_calamity, ["Hon'Ble Members Of Parliament"])

    # Aggregate calamity relief per MP per financial year
    calamity_agg = df2_calamity.groupby(["Hon'Ble Members Of Parliament"], as_index=False).agg(
        total_calamity_consented=('Consent Amount ( ₹ )', 'sum')
    )

    # Merge Allocated Funds with Calamity Consented
    mp_finances = pd.merge(
        df1_allocated, 
        calamity_agg, 
        on=["Hon'Ble Members Of Parliament"], 
        how='left'
    )
    mp_finances['total_calamity_consented'] = mp_finances['total_calamity_consented'].fillna(0.0)
    
    # Calculate Net Available Balance for the MP
    mp_finances['net_available_fund'] = mp_finances['Allocated Amount ( ₹ )'] - mp_finances['total_calamity_consented']

    # =========================================================================
    # 3. CLEAN WORK LIFECYCLE DATASETS (DATASETS 3, 4, 5 & 6)
    # Key: ['work_id']
    # =========================================================================
    # Dataset 3: Works Recommended
    df3_recommended = clean_currency_and_numbers(df3_recommended, ['Recommended Amount   ( ₹ )'])
    df3_recommended = clean_string_columns(df3_recommended, ['Work', "Hon'Ble Members Of Parliament", 'Work Description', 'Constituency', 'State','Work Category'])
    df3_recommended['Recommended Date'] = pd.to_datetime(df3_recommended['Recommended Date'], errors='coerce')
    df3_recommended['Sanction Date'] = pd.to_datetime(df3_recommended['Sanction Date'], errors='coerce')
    
    # Dataset 4: Works Sanctioned
    df4_sanctioned = clean_currency_and_numbers(df4_sanctioned, ['Sanction Amount ( ₹ )'])
    df4_sanctioned = clean_string_columns(df4_sanctioned, ['Work', 'Ida', "Hon'Ble Members Of Parliament", 'Work Description', 'Vendor Name', 'Constituency', 'State','Work Category', 'Work Status'])
    df4_sanctioned['Recommended Date'] = pd.to_datetime(df4_sanctioned['Recommended Date'], errors='coerce')
    df4_sanctioned['Sanction Date'] = pd.to_datetime(df4_sanctioned['Sanction Date'], errors='coerce')
    
    # Dataset 5: Works Completed
    df5_completed = clean_string_columns(df5_completed, ['Work', "Hon'Ble Members Of Parliament", 'Work Description', 'Constituency', 'State','Work Category'])
    df5_completed['Completion Date'] = pd.to_datetime(df5_completed['Completion Date'], errors='coerce')
    
    # Dataset 6: Expenditure on Completed and Ongoing Works
    df6_expenditure = clean_currency_and_numbers(df6_expenditure, ['Fund Disbursed Amount ( ₹ )'])
    df6_expenditure = clean_string_columns(df6_expenditure, ['State','Work', "Hon'Ble Members Of Parliament", 'Vendor Name', 'Payment Status'])
    df6_expenditure['Expenditure Date'] = pd.to_datetime(df6_expenditure['Expenditure Date'], errors='coerce')
    
    # Group Dataset 6 by work_id in case there are multiple payment installments
    expenditure_agg = df6_expenditure.groupby('Work', as_index=False).agg(
        total_expenditure_released=('Fund Disbursed Amount ( ₹ )', 'max'),
        vendor_names=('Vendor Name', lambda x: ', '.join(x.dropna().unique())),
        latest_payment_date=('Expenditure Date', 'max'),
        payment_status=('Payment Status', 'last')
        )
    
    print("[*] Joining Datasets into Work-Level Master Dataframe...")

    # =========================================================================
    # 4. JOIN WORK-LEVEL DATASETS (3 -> 4 -> 5 -> 6)
    # =========================================================================
    # Step A: Recommended + Sanctioned
    works_master = pd.merge(
            df3_recommended, 
            df4_sanctioned, 
            on='Work', 
            how='left',
            suffixes=('', '_sanctioned')
        )
    
    # Step B: + Completed
    works_master = pd.merge(
            works_master, 
            df5_completed[['Work', 'Completion Date', 'Image']], 
            on='Work', 
            how='left'
        )
    
    # Step C: + Expenditure
    works_master = pd.merge(
            works_master, 
            expenditure_agg, 
            on='Work', 
            how='left'
        )

    # =========================================================================
    # 5. JOIN WITH FINANCIAL ALLOCATIONS (DATASETS 1 & 2)
    # =========================================================================
    master_df = pd.merge(
            works_master, 
            mp_finances[["Hon'Ble Members Of Parliament", 'Allocated Amount ( ₹ )', 'total_calamity_consented', 'net_available_fund']], 
            on=["Hon'Ble Members Of Parliament"], 
            how='left'
        )
    
    print("[*] Calculating Derived Audit Indicators & Discrepancy Flags...")

    # =========================================================================
    # 6. FEATURE ENGINEERING & ANOMALY INDICATORS
    # =========================================================================
    # A. Status Tracking
    master_df['is_sanctioned'] = master_df['Sanction Date'].notna()
    master_df['is_completed'] = master_df['Completion Date'].notna()
    master_df['has_expenditure'] = master_df['total_expenditure_released'] > 0
    
    # B. Cost Variance (Sanctioned vs Expenditure)
    master_df['cost_variance'] = master_df['total_expenditure_released'] - master_df['Sanction Amount ( ₹ )']
    master_df['is_cost_overrun'] = master_df['cost_variance'] > 0
    
    # C. Days Taken To Sanction (Recommendation to Sanction)
    master_df['days_to_sanction'] = (master_df['Sanction Date'] - master_df['Recommended Date']).dt.days
    
    # D. Days Taken To Complete (Sanction to Completion)
    master_df['days_to_complete'] = (master_df['Completion Date'] - master_df['Sanction Date']).dt.days

    # E. Discrepancy Flag: Payment made on unsanctioned work
    master_df['flag_unsanctioned_payment'] = (~master_df['is_sanctioned']) & (master_df['has_expenditure'])
    
    # F. Discrepancy Flag: Marked completed without image
    master_df['flag_missing_completion_cert'] = (master_df['is_completed']) & (master_df['Image'].isna() | (master_df['Image'].str.strip() == ''))
    
    print("[+] Master Dataframe built successfully!")
    print(f"    • Total Master Records: {len(master_df):,}")
    print(f"    • Total Unique MPs: {master_df["Hon'Ble Members Of Parliament"].nunique()}")
    print(f"    • Total Unique Works: {master_df['Work'].nunique()}")
    
    return master_df

master_df = load_and_merge_mplads_datasets()

# Automatically select GPU if available, otherwise fallback to CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[*] Initializing NLP Engine on Device: {device.upper()}")

nlp_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2", device=device
)


def run_mplads_ai_audit(
    master_df: pd.DataFrame, similarity_threshold: float = 0.70
):
    df = master_df.copy()

    # =========================================================================
    # 1. ISOLATION FOREST: FINANCIAL ANOMALY ENGINE
    # =========================================================================
    print("\n[*] [Engine 1/2] Executing Isolation Forest Anomaly Detection...")

    df["Sanction Amount ( ₹ )"] = df["Sanction Amount ( ₹ )"].fillna(0.0)
    df["total_expenditure_released"] = df["total_expenditure_released"].fillna(
        0.0
    )
    df["cost_variance"] = df["cost_variance"].fillna(0.0)

    feature_cols = [
        "Sanction Amount ( ₹ )",
        "total_expenditure_released",
        "cost_variance",
    ]
    X = df[feature_cols]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    iso_forest = IsolationForest(
        contamination=0.15, random_state=42, n_jobs=-1
    )
    df["cost_anomaly_flag"] = iso_forest.fit_predict(X_scaled)

    raw_scores = iso_forest.decision_function(X_scaled)
    min_s, max_s = raw_scores.min(), raw_scores.max()
    df["cost_anomaly_risk_score"] = np.round(
        (1.0 - ((raw_scores - min_s) / (max_s - min_s + 1e-6))) * 100, 2
    )

    # =========================================================================
    # 2. NLP: GPU-ACCELERATED VECTORIZED DUPLICATE DETECTION
    # =========================================================================
    print(
        "[*] [Engine 2/2] Executing GPU-Accelerated NLP Similarity Search..."
    )

    df["Work Description"] = df["Work Description"].fillna("UNKNOWN WORK")
    df["Constituency"] = df["Constituency"].fillna("UNKNOWN DISTRICT")

    flagged_duplicates = []

    for district_name, group in df.groupby("Constituency"):
        num_records = len(group)
        if num_records < 2:
            continue

        work_names = group["Work Description"].tolist()
        work_ids = group["Work"].tolist()
        mp_ids = (
            group["Hon'Ble Members Of Parliament"].tolist()
            if "Hon'Ble Members Of Parliament" in group.columns
            else ["N/A"] * num_records
        )

        # Batch encode on GPU (batch_size 256 for fast parallel execution)
        embeddings = nlp_model.encode(
            work_names,
            convert_to_tensor=True,
            batch_size=256,
            show_progress_bar=False,
            device=device,
        )

        # Normalize embeddings on GPU tensor space
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        # Fast GPU Matrix Multiplication
        sim_matrix = torch.mm(embeddings, embeddings.T)

        # Extract upper triangle matches above the similarity threshold
        upper_tri = torch.triu(sim_matrix, diagonal=1)
        match_indices = (
            (upper_tri >= similarity_threshold).nonzero(as_tuple=False).cpu()
        )

        # Append flagged pairs
        for idx in match_indices:
            i, j = idx[0].item(), idx[1].item()
            text_sim = float(sim_matrix[i, j].item())
            dup_risk = text_sim * 100.0

            flagged_duplicates.append({
                "original_work_id": work_ids[i],
                "original_work_name": work_names[i],
                "flagged_work_id": work_ids[j],
                "flagged_work_name": work_names[j],
                "mp_id": mp_ids[i],
                "district": district_name,
                "text_similarity_pct": round(text_sim * 100, 2),
                "duplicate_risk_score": round(dup_risk, 2),
            })

        # Clear PyTorch GPU cache per district to maintain low VRAM consumption
        del embeddings, sim_matrix, upper_tri
        if device == "cuda":
            torch.cuda.empty_cache()

    duplicates_df = pd.DataFrame(flagged_duplicates)

    # =========================================================================
    # 3. COMPOSITE AUDIT SCORE CALCULATOR
    # =========================================================================
    if not duplicates_df.empty:
        dup_summary = (
            duplicates_df.groupby("flagged_work_id")["duplicate_risk_score"]
            .max()
            .reset_index()
        )
        df = pd.merge(
            df,
            dup_summary,
            left_on="Work",
            right_on="flagged_work_id",
            how="left",
        )
        df["duplicate_risk_score"] = df["duplicate_risk_score"].fillna(0.0)
        df.drop(columns=["flagged_work_id"], inplace=True, errors="ignore")
    else:
        df["duplicate_risk_score"] = 0.0

    df["composite_risk_score"] = np.round(
        (df["cost_anomaly_risk_score"] * 0.5)
        + (df["duplicate_risk_score"] * 0.5),
        2,
    )

    print("[+] GPU Audit Engine Execution Complete!")
    return df, duplicates_df

print(run_mplads_ai_audit(master_df))