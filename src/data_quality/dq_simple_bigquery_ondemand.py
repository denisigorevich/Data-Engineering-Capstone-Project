"""
Capstone Project — Data Quality Management (Simple BigQuery OnDemand)

This script demonstrates:
1. Direct BigQuery table scanning (no Dataplex entities required)
2. On-demand execution only
3. No result export - results visible only in Dataplex console
4. Simplest implementation for quick testing

Data Quality Rules Applied:
- cylinders is not null (Completeness)
- VIN is exactly 17 characters (Validity)
- VIN is unique across all records (Uniqueness)
"""

import logging
from google.cloud import dataplex_v1
from google.cloud.dataplex_v1.types import DataScan, DataQualitySpec, DataQualityRule, DataSource, Trigger

# --- CONFIGURATION ---
PROJECT_ID = "dt-denis-sandbox-dev"
REGION = "us-central1"
DATASET_ID = "denis_sandbox"  # Your BigQuery dataset
TABLE_ID = "vehicles_cleaned"  # Your BigQuery table
DATASCAN_ID = "vehicles-dq-simple-ondemand-scan"
DATASCAN_DISPLAY_NAME = "Vehicles DQ Simple OnDemand Scan"

logging.basicConfig(level=logging.INFO)
client = dataplex_v1.DataScanServiceClient()
parent_path = f"projects/{PROJECT_ID}/locations/{REGION}"
scan_path = f"{parent_path}/dataScans/{DATASCAN_ID}"

# BigQuery table resource name
bq_resource = f"//bigquery.googleapis.com/projects/{PROJECT_ID}/datasets/{DATASET_ID}/tables/{TABLE_ID}"

# --- DEFINE DATA QUALITY RULES ---
dq_rules = [
    # Rule 1: Non-null check on 'cylinders'
    DataQualityRule(
        non_null_expectation=DataQualityRule.NonNullExpectation(),
        column="cylinders",
        dimension="COMPLETENESS"
    ),

    # Rule 2: VIN must be 17 characters
    DataQualityRule(
        row_condition_expectation=DataQualityRule.RowConditionExpectation(
            sql_expression="LENGTH(VIN) = 17"
        ),
        column="VIN",
        dimension="VALIDITY"
    ),

    # Rule 3: VIN uniqueness
    DataQualityRule(
        table_condition_expectation=DataQualityRule.TableConditionExpectation(
            sql_expression="COUNT(DISTINCT VIN) = COUNT(*)"
        ),
        dimension="UNIQUENESS"
    )
]

dq_spec = DataQualitySpec(rules=dq_rules)

# --- CREATE DATA SOURCE FOR BIGQUERY ---
data_source = DataSource(
    resource=bq_resource  # Direct BigQuery table reference
)

# --- CREATE EXECUTION SPEC FOR ON-DEMAND ---
execution_spec = DataScan.ExecutionSpec(
    trigger=Trigger(
        on_demand=Trigger.OnDemand()  # For on-demand execution
    )
)

# --- CREATE DATA SCAN ---
data_scan = DataScan(
    data=data_source,
    data_quality_spec=dq_spec,
    display_name=DATASCAN_DISPLAY_NAME,
    execution_spec=execution_spec
)

# --- CREATE OR UPDATE SCAN ---
def main():
    try:
        # Try to get existing scan
        existing = client.get_data_scan(name=scan_path)
        logging.info(f"DataScan already exists: {existing.name}")
        logging.info("Skipping creation since scan already exists. Delete manually if you want to recreate.")

    except Exception as e:
        logging.info(f"DataScan doesn't exist, creating new one: {e}")
        
        # Create new scan
        create_request = dataplex_v1.CreateDataScanRequest(
            parent=parent_path,
            data_scan_id=DATASCAN_ID,
            data_scan=data_scan
        )
        operation = client.create_data_scan(request=create_request)
        result = operation.result()
        logging.info(f"Created new DataScan: {result.name}")

    # --- RUN SCAN ON DEMAND ---
    try:
        run_request = dataplex_v1.RunDataScanRequest(name=scan_path)
        run_response = client.run_data_scan(request=run_request)
        logging.info(f"Triggered scan job: {run_response.job.name}")
    except Exception as e:
        logging.error(f"Failed to run scan: {e}")

if __name__ == "__main__":
    main() 
