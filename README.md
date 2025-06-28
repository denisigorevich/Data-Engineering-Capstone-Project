# Cloud Data Platform – Capstone Project (CarClassifieds)

**Goal:** deliver an end-to-end GCP data platform that ingests raw vehicle listings,  
cleans & profiles them, enforces data-quality SLAs, enriches with the NHTSA vPIC API,  
and serves analytics & ML use-cases … all inside `europe-west2`.

## Repository layout

capstone-cloud-data-platform/          # Git repository root
│
├── README.md                          # ONE root read-me (see template below)
├── .gitignore
│
├── infra/                             # Terraform, Cloud Build YAML, etc. (later phases)
│
├── data_ingestion/                    # Phase 1  – raw ingestion
│   └── clean_vehicles.py              # idempotent ingestion + cleaning CLI
│
├── data_quality/                      # Phase 2  – profiling & rules
│   ├── dq_basic_dataplex_entity.py    # Dataplex entity scan (learning/demo)
│   ├── dq_simple_bigquery_ondemand.py # On-demand BigQuery scan (fast test)
│   ├── dq_scheduled_with_export.py    # Scheduled scan + BQ export (prod)
│   ├── test_dataplex_connection.py    # smoke-test / permissions
│   └── README.md
│
├── docs/                              # Design docs, diagrams, ad-hoc reports
│   ├── data_profiling_report.md       # one-time profiling findings
│   └── hld_architecture.drawio        # Lucidchart or Draw.io export
│
├── requirements.txt                   # shared, pinned versions
└── .github/workflows/…                # CI lint / unit-tests (optional)


## Quick-start

```bash
# 1️⃣  Create and activate virtual-env
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2️⃣  Ingest & clean raw CSV
python data_ingestion/clean_vehicles.py --upload-to-gcs --load-to-bigquery

# 3️⃣  Run basic on-demand quality scan
python data_quality/dq_simple_bigquery_ondemand.py
