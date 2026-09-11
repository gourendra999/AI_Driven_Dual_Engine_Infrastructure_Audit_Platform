# MPLADS AI Audit — Streamlit UI

## Local project structure

```text
project/
├── app.py
├── requirements.txt
├── Procfile
├── assets/
│   ├── Allocated Limit for Honble MPs.csv
│   ├── Amount consented for Calamity.csv
│   ├── Works Recommended.csv
│   ├── Works Sanctioned.csv
│   ├── Works Completed.csv
│   └── Expenditure on Completed and On-going Works as on Date.csv
└── .streamlit/
    └── config.toml
```

The app also accepts the original `assests/` spelling for compatibility with your current code.

## Local run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Render

Build command:

```bash
pip install -r requirements.txt
```

Start command:

```bash
streamlit run app.py --server.port $PORT --server.address 0.0.0.0
```

The `Procfile` is included as an alternative.

## Important

`sentence-transformers` downloads the MiniLM model the first time it runs. On CPU, NLP duplicate detection can be slow for very large datasets. GPU is used automatically when CUDA is available.
