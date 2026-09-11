import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# ============================================================
# PAGE / THEME
# ============================================================
st.set_page_config(
    page_title="MPLADS AI Audit",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main { background: #f5f7fb; }
.block-container { padding: 1.25rem 2rem 2rem; max-width: 1600px; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0b1730 0%, #132746 100%);
    border-right: 1px solid rgba(255,255,255,.08);
}
[data-testid="stSidebar"] * { color: #eef4ff !important; }

.brand { padding: 4px 0 20px; border-bottom: 1px solid rgba(255,255,255,.12); margin-bottom: 18px; }
.brand-title { font-size: 20px; font-weight: 800; letter-spacing: -.4px; }
.brand-sub { font-size: 11px; color: #a9b7cf !important; margin-top: 4px; }

.hero {
    background: linear-gradient(135deg, #ffffff 0%, #f7fbff 100%);
    border: 1px solid #e5eaf2;
    border-radius: 18px;
    padding: 22px 24px;
    box-shadow: 0 8px 28px rgba(20,35,65,.05);
}
.hero-title { font-size: 30px; font-weight: 800; color: #10203c; margin-bottom: 3px; }
.hero-sub { color: #6b7890; font-size: 14px; }

.kpi {
    background: #fff;
    border: 1px solid #e6ebf2;
    border-radius: 16px;
    padding: 16px 18px;
    min-height: 120px;
    box-shadow: 0 7px 22px rgba(20,35,65,.04);
}
.kpi-label { color:#6b7890; font-size:12px; font-weight:600; }
.kpi-value { color:#12223f; font-size:28px; font-weight:800; margin-top:5px; }
.kpi-help { color:#8a95a8; font-size:11px; margin-top:4px; }

.card {
    background:#fff;
    border:1px solid #e5eaf2;
    border-radius:16px;
    padding:12px 16px 16px;
    box-shadow:0 7px 22px rgba(20,35,65,.04);
    margin-bottom: 14px;
}
.card-title { font-size:14px; font-weight:750; color:#172844; margin-bottom:2px; }
.card-sub { font-size:11px; color:#8190a7; margin-bottom:8px; }

.badge { display:inline-block; border-radius:99px; padding:4px 8px; font-size:10px; font-weight:700; }
.badge-red { background:#fff0f1; color:#d83a45; }
.badge-orange { background:#fff5e9; color:#c86e13; }
.badge-green { background:#ecf9f1; color:#15915b; }

[data-testid="stMetricValue"] { color: #11213d; }
[data-testid="stMetricLabel"] { color: #68778e; }

button[kind="primary"] {
    border-radius: 10px !important;
    background: linear-gradient(90deg,#2167e8,#3c84ff) !important;
}

.small-note { color:#7f8ba0; font-size:11px; }
.footer { color:#8290a4; font-size:11px; text-align:center; padding:18px 0 4px; }
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
# CONSTANTS
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
ASSET_CANDIDATES = [BASE_DIR / "assets", BASE_DIR / "assests"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

REQUIRED_FILES = [
    "Allocated Limit for Honble MPs.csv",
    "Amount consented for Calamity.csv",
    "Works Recommended.csv",
    "Works Sanctioned.csv",
    "Works Completed.csv",
    "Expenditure on Completed and On-going Works as on Date.csv",
]

# ============================================================
# HELPERS
# ============================================================
def asset_dir() -> Path:
    for p in ASSET_CANDIDATES:
        if p.exists():
            return p
    return BASE_DIR / "assets"


def fmt_inr(value: float) -> str:
    if pd.isna(value):
        value = 0
    value = float(value)
    if abs(value) >= 1e7:
        return f"₹{value/1e7:.2f} Cr"
    if abs(value) >= 1e5:
        return f"₹{value/1e5:.2f} L"
    return f"₹{value:,.0f}"


def clean_currency_and_numbers(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(r"[₹,]", "", regex=True)
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return df


def clean_string_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.upper()
    return df


def normalize_colnames(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip().str.title()
    return df.drop(columns=["Sr. No."], errors="ignore")


def choose_col(df: pd.DataFrame, names: List[str]) -> str | None:
    for n in names:
        if n in df.columns:
            return n
    return None


@st.cache_data(show_spinner=False)
def load_and_merge_mplads_datasets_from_files(file_map: Dict[str, bytes | str]) -> pd.DataFrame:
    def read_csv_any(source):
        return pd.read_csv(source)

    df1 = normalize_colnames(read_csv_any(file_map[REQUIRED_FILES[0]]))
    df2 = normalize_colnames(read_csv_any(file_map[REQUIRED_FILES[1]]))
    df3 = normalize_colnames(read_csv_any(file_map[REQUIRED_FILES[2]]))
    df4 = normalize_colnames(read_csv_any(file_map[REQUIRED_FILES[3]]))
    df5 = normalize_colnames(read_csv_any(file_map[REQUIRED_FILES[4]]))
    df6 = normalize_colnames(read_csv_any(file_map[REQUIRED_FILES[5]]))

    df1 = df1.rename(columns={"Hon'Ble Members Of Parliaments": "Hon'Ble Members Of Parliament"})

    if "Work Id" in df6.columns and "Work" in df6.columns:
        df6["Work"] = df6["Work Id"].fillna("").astype(str) + "-" + df6["Work"].fillna("").astype(str)

    df1 = clean_currency_and_numbers(df1, ["Allocated Amount ( ₹ )"])
    df1 = clean_string_columns(df1, ["Hon'Ble Members Of Parliament", "Constituency"])

    df2 = clean_currency_and_numbers(df2, ["Consent Amount ( ₹ )"])
    df2 = clean_string_columns(df2, ["Hon'Ble Members Of Parliament"])
    calamity_agg = df2.groupby(["Hon'Ble Members Of Parliament"], as_index=False).agg(
        total_calamity_consented=("Consent Amount ( ₹ )", "sum")
    )

    mp_finances = pd.merge(df1, calamity_agg, on=["Hon'Ble Members Of Parliament"], how="left")
    mp_finances["total_calamity_consented"] = mp_finances["total_calamity_consented"].fillna(0.0)
    mp_finances["net_available_fund"] = (
        mp_finances["Allocated Amount ( ₹ )"] - mp_finances["total_calamity_consented"]
    )

    df3 = clean_currency_and_numbers(df3, ["Recommended Amount   ( ₹ )"])
    df3 = clean_string_columns(
        df3, ["Work", "Hon'Ble Members Of Parliament", "Work Description", "Constituency", "State", "Work Category"]
    )
    for c in ["Recommended Date", "Sanction Date"]:
        if c in df3.columns:
            df3[c] = pd.to_datetime(df3[c], errors="coerce")

    df4 = clean_currency_and_numbers(df4, ["Sanction Amount ( ₹ )"])
    df4 = clean_string_columns(
        df4,
        ["Work", "Ida", "Hon'Ble Members Of Parliament", "Work Description", "Vendor Name", "Constituency", "State", "Work Category", "Work Status"],
    )
    for c in ["Recommended Date", "Sanction Date"]:
        if c in df4.columns:
            df4[c] = pd.to_datetime(df4[c], errors="coerce")

    df5 = clean_string_columns(
        df5, ["Work", "Hon'Ble Members Of Parliament", "Work Description", "Constituency", "State", "Work Category"]
    )
    if "Completion Date" in df5.columns:
        df5["Completion Date"] = pd.to_datetime(df5["Completion Date"], errors="coerce")

    df6 = clean_currency_and_numbers(df6, ["Fund Disbursed Amount ( ₹ )"])
    df6 = clean_string_columns(df6, ["State", "Work", "Hon'Ble Members Of Parliament", "Vendor Name", "Payment Status"])
    if "Expenditure Date" in df6.columns:
        df6["Expenditure Date"] = pd.to_datetime(df6["Expenditure Date"], errors="coerce")

    expenditure_agg = df6.groupby("Work", as_index=False).agg(
        total_expenditure_released=("Fund Disbursed Amount ( ₹ )", "max"),
        vendor_names=("Vendor Name", lambda x: ", ".join(x.dropna().astype(str).unique())),
        latest_payment_date=("Expenditure Date", "max"),
        payment_status=("Payment Status", "last"),
    )

    works_master = pd.merge(df3, df4, on="Work", how="left", suffixes=("", "_sanctioned"))
    keep5 = [c for c in ["Work", "Completion Date", "Image"] if c in df5.columns]
    if keep5:
        works_master = pd.merge(works_master, df5[keep5], on="Work", how="left")
    works_master = pd.merge(works_master, expenditure_agg, on="Work", how="left")

    master_df = pd.merge(
        works_master,
        mp_finances[["Hon'Ble Members Of Parliament", "Allocated Amount ( ₹ )", "total_calamity_consented", "net_available_fund"]],
        on=["Hon'Ble Members Of Parliament"],
        how="left",
    )

    master_df["is_sanctioned"] = master_df["Sanction Date"].notna()
    master_df["is_completed"] = master_df["Completion Date"].notna()
    master_df["has_expenditure"] = master_df["total_expenditure_released"].fillna(0) > 0
    master_df["cost_variance"] = master_df["total_expenditure_released"].fillna(0) - master_df["Sanction Amount ( ₹ )"].fillna(0)
    master_df["is_cost_overrun"] = master_df["cost_variance"] > 0
    master_df["days_to_sanction"] = (
        master_df["Sanction Date"] - master_df["Recommended Date"]
    ).dt.days
    master_df["days_to_complete"] = (
        master_df["Completion Date"] - master_df["Sanction Date"]
    ).dt.days
    master_df["flag_unsanctioned_payment"] = (~master_df["is_sanctioned"]) & master_df["has_expenditure"]

    image_series = master_df["Image"].fillna("").astype(str).str.strip() if "Image" in master_df.columns else pd.Series("", index=master_df.index)
    master_df["flag_missing_completion_cert"] = master_df["is_completed"] & (image_series == "")

    return master_df


@st.cache_resource(show_spinner=False)
def get_nlp_model(device: str):
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)


def run_mplads_ai_audit(master_df: pd.DataFrame, similarity_threshold: float = 0.70):
    df = master_df.copy()

    for c in ["Sanction Amount ( ₹ )", "total_expenditure_released", "cost_variance"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    X = df[["Sanction Amount ( ₹ )", "total_expenditure_released", "cost_variance"]]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    iso_forest = IsolationForest(contamination=0.15, random_state=42, n_jobs=-1)
    df["cost_anomaly_flag"] = iso_forest.fit_predict(X_scaled)
    raw_scores = iso_forest.decision_function(X_scaled)
    min_s, max_s = raw_scores.min(), raw_scores.max()
    df["cost_anomaly_risk_score"] = np.round(
        (1.0 - ((raw_scores - min_s) / (max_s - min_s + 1e-6))) * 100, 2
    )

    model = get_nlp_model(DEVICE)
    df["Work Description"] = df["Work Description"].fillna("UNKNOWN WORK")
    df["Constituency"] = df["Constituency"].fillna("UNKNOWN DISTRICT")

    flagged_duplicates = []
    for district_name, group in df.groupby("Constituency"):
        if len(group) < 2:
            continue

        work_names = group["Work Description"].tolist()
        work_ids = group["Work"].tolist()
        mp_ids = group["Hon'Ble Members Of Parliament"].tolist()

        embeddings = model.encode(
            work_names,
            convert_to_tensor=True,
            batch_size=256,
            show_progress_bar=False,
            device=DEVICE,
        )
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        sim_matrix = torch.mm(embeddings, embeddings.T)
        upper_tri = torch.triu(sim_matrix, diagonal=1)
        match_indices = (upper_tri >= similarity_threshold).nonzero(as_tuple=False).cpu()

        for idx in match_indices:
            i, j = idx[0].item(), idx[1].item()
            text_sim = float(sim_matrix[i, j].item())
            flagged_duplicates.append(
                {
                    "original_work_id": work_ids[i],
                    "original_work_name": work_names[i],
                    "flagged_work_id": work_ids[j],
                    "flagged_work_name": work_names[j],
                    "mp_id": mp_ids[i],
                    "district": district_name,
                    "text_similarity_pct": round(text_sim * 100, 2),
                    "duplicate_risk_score": round(text_sim * 100, 2),
                }
            )

        del embeddings, sim_matrix, upper_tri
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    duplicates_df = pd.DataFrame(flagged_duplicates)

    if not duplicates_df.empty:
        dup_summary = duplicates_df.groupby("flagged_work_id")["duplicate_risk_score"].max().reset_index()
        df = pd.merge(df, dup_summary, left_on="Work", right_on="flagged_work_id", how="left")
        df["duplicate_risk_score"] = df["duplicate_risk_score"].fillna(0.0)
        df.drop(columns=["flagged_work_id"], inplace=True, errors="ignore")
    else:
        df["duplicate_risk_score"] = 0.0

    df["composite_risk_score"] = np.round(
        (df["cost_anomaly_risk_score"] * 0.5) + (df["duplicate_risk_score"] * 0.5), 2
    )
    return df, duplicates_df


# ============================================================
# FILE SOURCING
# ============================================================
def get_default_file_map() -> Dict[str, str] | None:
    base = asset_dir()
    found = {name: str(base / name) for name in REQUIRED_FILES if (base / name).exists()}
    return found if len(found) == len(REQUIRED_FILES) else None


# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.markdown(
    """
<div class='brand'>
  <div class='brand-title'>🏛️ MPLADS AI Audit</div>
  <div class='brand-sub'>Transparent Funds &nbsp;|&nbsp; Better Infrastructure</div>
</div>
""",
    unsafe_allow_html=True,
)

page = st.sidebar.radio(
    "NAVIGATION",
    ["Dashboard", "MP Overview", "Works Analysis", "Financials", "Audit Flags", "Anomaly Detection", "Duplicate Detection", "Reports", "Settings"],
    label_visibility="visible",
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Engine:** `{DEVICE.upper()}`")
st.sidebar.markdown("<span class='small-note'>SentenceTransformer + Isolation Forest</span>", unsafe_allow_html=True)

# Data source controls
st.sidebar.markdown("### Data Source")
default_files = get_default_file_map()
use_uploads = st.sidebar.toggle("Upload CSVs", value=False)

file_map = None
if use_uploads:
    uploaded = {}
    for name in REQUIRED_FILES:
        f = st.sidebar.file_uploader(name, type=["csv"], key=f"up_{name}")
        if f is not None:
            uploaded[name] = f.getvalue()
    if len(uploaded) == len(REQUIRED_FILES):
        file_map = uploaded
else:
    file_map = default_files

similarity_threshold = st.sidebar.slider(
    "Duplicate similarity threshold",
    min_value=0.50,
    max_value=0.95,
    value=0.70,
    step=0.01,
)

run_clicked = st.sidebar.button("↻  Run New Audit", type="primary", use_container_width=True)

# ============================================================
# LOAD DATA
# ============================================================
if file_map is None:
    st.markdown(
        """
        <div class='hero'>
          <div class='hero-title'>MPLADS AI Audit</div>
          <div class='hero-sub'>Your dashboard is ready. Add the six MPLADS CSV files in an <b>assets/</b> or <b>assests/</b> folder, or enable CSV upload from the sidebar.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.info("Required files: " + " • ".join(REQUIRED_FILES))
    st.stop()

try:
    with st.spinner("Loading and merging six MPLADS datasets..."):
        master_df = load_and_merge_mplads_datasets_from_files(file_map)
except Exception as exc:
    st.error(f"Dataset loading failed: {exc}")
    st.stop()

cache_key = (similarity_threshold, tuple(sorted(master_df.shape)))
if run_clicked or "audit_df" not in st.session_state or st.session_state.get("audit_key") != cache_key:
    with st.spinner("Running Isolation Forest + NLP duplicate detection..."):
        audit_df, duplicates_df = run_mplads_ai_audit(master_df, similarity_threshold)
        st.session_state.audit_df = audit_df
        st.session_state.duplicates_df = duplicates_df
        st.session_state.audit_key = cache_key
else:
    audit_df = st.session_state.audit_df
    duplicates_df = st.session_state.duplicates_df

# ============================================================
# DERIVED METRICS
# ============================================================
total_records = len(audit_df)
unique_mps = audit_df["Hon'Ble Members Of Parliament"].nunique()
unique_works = audit_df["Work"].nunique()
flagged_mask = (audit_df["composite_risk_score"] >= 70) | audit_df["flag_unsanctioned_payment"] | audit_df["flag_missing_completion_cert"]
flagged_works = int(flagged_mask.sum())

overall_risk = float(audit_df["composite_risk_score"].mean()) if total_records else 0.0
high_risk = int((audit_df["composite_risk_score"] >= 70).sum())
medium_risk = int(((audit_df["composite_risk_score"] >= 40) & (audit_df["composite_risk_score"] < 70)).sum())
low_risk = int((audit_df["composite_risk_score"] < 40).sum())
financial_anomalies = int((audit_df["cost_anomaly_flag"] == -1).sum())
duplicate_pairs = len(duplicates_df)
cost_overruns = int(audit_df["is_cost_overrun"].sum())
unsanctioned_payments = int(audit_df["flag_unsanctioned_payment"].sum())
missing_cert = int(audit_df["flag_missing_completion_cert"].sum())

# ============================================================
# HEADER
# ============================================================
col_a, col_b = st.columns([7, 2])
with col_a:
    st.markdown(
        """
<div class='hero'>
  <div class='hero-title'>AI-Powered Audit Dashboard</div>
  <div class='hero-sub'>Multi-dataset analysis, financial anomaly detection and duplicate-work discovery for MPLADS works.</div>
</div>
""",
        unsafe_allow_html=True,
    )
with col_b:
    st.metric("Audit Engine", DEVICE.upper(), help="GPU is automatically selected when CUDA is available.")

st.write("")

# ============================================================
# KPI ROW
# ============================================================
kpis = [
    ("Total Records", f"{total_records:,}", "Work-level master records"),
    ("Unique MPs", f"{unique_mps:,}", "Members represented in dataset"),
    ("Unique Works", f"{unique_works:,}", "Distinct work IDs"),
    ("Flagged Works", f"{flagged_works:,}", "Potential audit concerns"),
]
cols = st.columns(4)
for c, (label, value, help_text) in zip(cols, kpis):
    with c:
        st.markdown(
            f"<div class='kpi'><div class='kpi-label'>{label}</div><div class='kpi-value'>{value}</div><div class='kpi-help'>{help_text}</div></div>",
            unsafe_allow_html=True,
        )

st.write("")

# ============================================================
# FILTERS
# ============================================================
with st.container(border=True):
    st.markdown("**Dashboard Filters**")
    f1, f2, f3, f4 = st.columns(4)
    states = sorted(audit_df["State"].dropna().astype(str).unique()) if "State" in audit_df.columns else []
    districts = sorted(audit_df["Constituency"].dropna().astype(str).unique()) if "Constituency" in audit_df.columns else []
    risk_options = ["All", "High Risk", "Medium Risk", "Low Risk"]
    with f1:
        selected_state = st.selectbox("State", ["All"] + states)
    with f2:
        selected_district = st.selectbox("Constituency / District", ["All"] + districts)
    with f3:
        selected_risk = st.selectbox("Risk Band", risk_options)
    with f4:
        min_score, max_score = st.slider("Composite Score", 0, 100, (0, 100))

filtered_df = audit_df.copy()
if selected_state != "All":
    filtered_df = filtered_df[filtered_df["State"] == selected_state]
if selected_district != "All":
    filtered_df = filtered_df[filtered_df["Constituency"] == selected_district]
if selected_risk == "High Risk":
    filtered_df = filtered_df[filtered_df["composite_risk_score"] >= 70]
elif selected_risk == "Medium Risk":
    filtered_df = filtered_df[filtered_df["composite_risk_score"].between(40, 69.999)]
elif selected_risk == "Low Risk":
    filtered_df = filtered_df[filtered_df["composite_risk_score"] < 40]
filtered_df = filtered_df[filtered_df["composite_risk_score"].between(min_score, max_score)]

# ============================================================
# PAGE CONTENT
# ============================================================
if page == "Dashboard":
    c1, c2, c3 = st.columns([1, 1, 1])

    with c1:
        st.markdown("<div class='card'><div class='card-title'>Financial Anomalies</div><div class='card-sub'>Isolation Forest risk distribution</div>", unsafe_allow_html=True)
        fig = go.Figure(go.Pie(
            labels=["High", "Medium", "Low"],
            values=[high_risk, medium_risk, low_risk],
            hole=.62,
            textinfo="percent",
        ))
        fig.update_layout(height=250, margin=dict(l=0,r=0,t=5,b=0), showlegend=True, legend=dict(orientation="h", y=-0.08))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown(f"**{financial_anomalies:,}** financial records classified as Isolation Forest anomalies.", unsafe_allow_html=False)
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown("<div class='card'><div class='card-title'>Duplicate Detection</div><div class='card-sub'>NLP similarity matches above threshold</div>", unsafe_allow_html=True)
        if duplicate_pairs:
            dup_bucket = pd.cut(
                duplicates_df["text_similarity_pct"], bins=[0, 70, 80, 90, 100], labels=["50–69%", "70–79%", "80–89%", "90%+"]
            )
            counts = dup_bucket.value_counts().reindex(["90%+", "80–89%", "70–79%", "50–69%"], fill_value=0)
            fig = go.Figure(go.Pie(labels=counts.index, values=counts.values, hole=.62, textinfo="percent"))
            fig.update_layout(height=250, margin=dict(l=0,r=0,t=5,b=0), showlegend=True, legend=dict(orientation="h", y=-0.08))
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            st.markdown(f"**{duplicate_pairs:,}** potential duplicate pairs detected.")
        else:
            st.info("No duplicate pairs crossed the current similarity threshold.")
        st.markdown("</div>", unsafe_allow_html=True)

    with c3:
        st.markdown("<div class='card'><div class='card-title'>Composite Risk Score</div><div class='card-sub'>50% financial anomaly + 50% duplicate risk</div>", unsafe_allow_html=True)
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=overall_risk,
            number={"suffix": " / 100", "font": {"size": 34}},
            gauge={"axis": {"range": [0,100]}, "bar": {"thickness": .25}, "steps": [
                {"range": [0,40], "color": "#e8f7ef"},
                {"range": [40,70], "color": "#fff4df"},
                {"range": [70,100], "color": "#ffe9eb"},
            ]},
        ))
        gauge.update_layout(height=235, margin=dict(l=10,r=10,t=5,b=0))
        st.plotly_chart(gauge, use_container_width=True, config={"displayModeBar": False})
        risk_text = "High Risk" if overall_risk >= 70 else "Moderate Risk" if overall_risk >= 40 else "Low Risk"
        st.markdown(f"**{risk_text}** &nbsp; · &nbsp; {high_risk:,} high-risk works")
        st.markdown("</div>", unsafe_allow_html=True)

    c4, c5, c6 = st.columns([1.2, 1, .8])
    with c4:
        st.markdown("<div class='card'><div class='card-title'>Top Constituencies by Risk</div><div class='card-sub'>Mean composite risk score</div>", unsafe_allow_html=True)
        top_dist = (
            filtered_df.groupby("Constituency", as_index=False)["composite_risk_score"].mean()
            .sort_values("composite_risk_score", ascending=False).head(8)
        )
        if not top_dist.empty:
            fig = px.bar(top_dist.sort_values("composite_risk_score"), x="composite_risk_score", y="Constituency", orientation="h", text_auto=".1f")
            fig.update_layout(height=330, margin=dict(l=10,r=10,t=5,b=10), xaxis_title=None, yaxis_title=None)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("No data for the selected filters.")
        st.markdown("</div>", unsafe_allow_html=True)

    with c5:
        st.markdown("<div class='card'><div class='card-title'>Risk Distribution</div><div class='card-sub'>Filtered dataset</div>", unsafe_allow_html=True)
        hist = px.histogram(filtered_df, x="composite_risk_score", nbins=20)
        hist.update_layout(height=330, margin=dict(l=10,r=10,t=5,b=10), xaxis_title="Composite Risk Score", yaxis_title="Works")
        st.plotly_chart(hist, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    with c6:
        st.markdown("<div class='card'><div class='card-title'>Key Insights</div><div class='card-sub'>Audit signals worth reviewing</div>", unsafe_allow_html=True)
        insight_items = [
            ("⚠️", f"{high_risk:,} works have high composite risk."),
            ("🔗", f"{duplicate_pairs:,} possible duplicate pairs detected."),
            ("💰", f"{cost_overruns:,} works show expenditure above sanction."),
            ("📷", f"{missing_cert:,} completed works lack an image/certificate value."),
            ("🧾", f"{unsanctioned_payments:,} works show expenditure without a sanction date."),
        ]
        for icon, text in insight_items:
            st.markdown(f"<div style='margin:10px 0;font-size:12px'>{icon}&nbsp; {text}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'><div class='card-title'>Flagged Works</div><div class='card-sub'>Highest-risk records in the current filter</div>", unsafe_allow_html=True)
    view = filtered_df.copy()
    view["flag"] = np.select(
        [view["flag_unsanctioned_payment"], view["flag_missing_completion_cert"], view["is_cost_overrun"], view["duplicate_risk_score"] >= similarity_threshold*100],
        ["Unsanctioned Payment", "Missing Completion Evidence", "Cost Overrun", "Duplicate"],
        default="Review",
    )
    view = view.sort_values("composite_risk_score", ascending=False).head(15)
    cols_show = [
        "Work", "Work Description", "Hon'Ble Members Of Parliament", "Constituency",
        "Sanction Amount ( ₹ )", "total_expenditure_released", "cost_variance", "composite_risk_score", "flag"
    ]
    cols_show = [c for c in cols_show if c in view.columns]
    st.dataframe(view[cols_show], use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

elif page == "MP Overview":
    st.subheader("MP Overview")
    mp = filtered_df.groupby("Hon'Ble Members Of Parliament", as_index=False).agg(
        works=("Work", "nunique"),
        allocated=("Allocated Amount ( ₹ )", "max"),
        calamity=("total_calamity_consented", "max"),
        expenditure=("total_expenditure_released", "sum"),
        avg_risk=("composite_risk_score", "mean"),
    ).sort_values("avg_risk", ascending=False)
    st.dataframe(mp, use_container_width=True, hide_index=True)
    fig = px.bar(mp.head(15).sort_values("avg_risk"), x="avg_risk", y="Hon'Ble Members Of Parliament", orientation="h", text_auto=".1f")
    fig.update_layout(height=500, xaxis_title="Average Composite Risk", yaxis_title=None)
    st.plotly_chart(fig, use_container_width=True)

elif page == "Works Analysis":
    st.subheader("Works Analysis")
    a, b, c = st.columns(3)
    with a: st.metric("Recommended", f"{len(filtered_df):,}")
    with b: st.metric("Sanctioned", f"{filtered_df['is_sanctioned'].sum():,}")
    with c: st.metric("Completed", f"{filtered_df['is_completed'].sum():,}")

    cat_col = "Work Category" if "Work Category" in filtered_df.columns else None
    if cat_col:
        cat = filtered_df.groupby(cat_col).agg(works=("Work","nunique"), risk=("composite_risk_score","mean")).reset_index().sort_values("works", ascending=False).head(15)
        fig = px.bar(cat, x="works", y=cat_col, orientation="h", text_auto=True)
        fig.update_layout(height=500, xaxis_title="Unique Works", yaxis_title=None)
        st.plotly_chart(fig, use_container_width=True)
    st.dataframe(filtered_df.sort_values("composite_risk_score", ascending=False), use_container_width=True, hide_index=True)

elif page == "Financials":
    st.subheader("Financials")
    total_alloc = filtered_df["Allocated Amount ( ₹ )"].sum()
    total_calamity = filtered_df["total_calamity_consented"].sum()
    total_sanction = filtered_df["Sanction Amount ( ₹ )"].sum()
    total_exp = filtered_df["total_expenditure_released"].sum()
    cols = st.columns(4)
    for c, label, value in zip(cols, ["Allocated", "Calamity Consented", "Sanctioned", "Expenditure"], [total_alloc, total_calamity, total_sanction, total_exp]):
        with c: st.metric(label, fmt_inr(value))

    fin = pd.DataFrame({"Metric": ["Allocated", "Calamity", "Sanctioned", "Expenditure"], "Amount": [total_alloc,total_calamity,total_sanction,total_exp]})
    fig = px.bar(fin, x="Metric", y="Amount", text_auto=".2s")
    fig.update_layout(height=400, yaxis_title="Amount (₹)")
    st.plotly_chart(fig, use_container_width=True)

elif page == "Audit Flags":
    st.subheader("Audit Flags")
    flags = pd.DataFrame({
        "Flag": ["Cost Overrun", "Unsanctioned Payment", "Missing Completion Evidence", "Financial Anomaly", "Potential Duplicate"],
        "Count": [cost_overruns, unsanctioned_payments, missing_cert, financial_anomalies, duplicate_pairs],
    })
    fig = px.bar(flags.sort_values("Count"), x="Count", y="Flag", orientation="h", text_auto=True)
    fig.update_layout(height=370)
    st.plotly_chart(fig, use_container_width=True)

    flagged = filtered_df[(filtered_df["composite_risk_score"] >= 70) | filtered_df["flag_unsanctioned_payment"] | filtered_df["flag_missing_completion_cert"]].copy()
    st.dataframe(flagged.sort_values("composite_risk_score", ascending=False), use_container_width=True, hide_index=True)

elif page == "Anomaly Detection":
    st.subheader("Financial Anomaly Detection")
    anomaly_df = filtered_df[filtered_df["cost_anomaly_flag"] == -1].copy().sort_values("cost_anomaly_risk_score", ascending=False)
    st.write(f"Isolation Forest flagged **{len(anomaly_df):,}** records in the current filter.")
    st.dataframe(anomaly_df[[c for c in ["Work","Work Description","Sanction Amount ( ₹ )","total_expenditure_released","cost_variance","cost_anomaly_risk_score","composite_risk_score"] if c in anomaly_df.columns]], use_container_width=True, hide_index=True)

elif page == "Duplicate Detection":
    st.subheader("NLP Duplicate Detection")
    if duplicates_df.empty:
        st.success("No potential duplicate pairs detected at the selected similarity threshold.")
    else:
        dup_view = duplicates_df.sort_values("duplicate_risk_score", ascending=False)
        st.metric("Potential Duplicate Pairs", f"{len(dup_view):,}")
        st.dataframe(dup_view, use_container_width=True, hide_index=True)
        top_districts = dup_view.groupby("district").size().reset_index(name="pairs").sort_values("pairs", ascending=False).head(15)
        fig = px.bar(top_districts.sort_values("pairs"), x="pairs", y="district", orientation="h", text_auto=True)
        fig.update_layout(height=450, xaxis_title="Duplicate Pairs", yaxis_title=None)
        st.plotly_chart(fig, use_container_width=True)

elif page == "Reports":
    st.subheader("Audit Reports")
    report_df = filtered_df.sort_values("composite_risk_score", ascending=False).copy()
    csv_bytes = report_df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇ Download Filtered Audit CSV", csv_bytes, "mplads_audit_report.csv", "text/csv", use_container_width=False)

    summary = {
        "Total records": total_records,
        "Unique MPs": unique_mps,
        "Unique works": unique_works,
        "High risk works": high_risk,
        "Financial anomalies": financial_anomalies,
        "Duplicate pairs": duplicate_pairs,
        "Cost overruns": cost_overruns,
        "Unsanctioned payments": unsanctioned_payments,
        "Missing completion evidence": missing_cert,
        "Overall composite risk": round(overall_risk, 2),
        "Device": DEVICE.upper(),
    }
    st.json(summary)

elif page == "Settings":
    st.subheader("Settings")
    st.write("**NLP model:** `sentence-transformers/all-MiniLM-L6-v2`")
    st.write(f"**Compute device:** `{DEVICE.upper()}`")
    st.write(f"**Similarity threshold:** `{similarity_threshold:.2f}`")
    st.write("**Isolation Forest contamination:** `0.15`")
    st.info("For production deployment with large datasets, consider adding approximate nearest-neighbor search or restricting duplicate comparisons by work category/date to reduce O(n²) comparisons within districts.")

st.markdown("<div class='footer'>MPLADS AI Audit · Data quality and anomaly signals are decision-support indicators, not final audit findings.</div>", unsafe_allow_html=True)
