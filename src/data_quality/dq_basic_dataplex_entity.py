"""
Capstone Project — Data Quality Management (Basic Dataplex Entity)

This script demonstrates:
1. Using Dataplex entities (lakes/zones/entities) as data source
2. Basic on-demand data quality scanning
3. Simple rule implementation without export features

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
LAKE_ID = "vehicle-lake"
ZONE_ID = "curated-zone"
ENTITY_ID = "vehicles_cleaned"
DATASCAN_ID = "vehicles-dq-basic-entity-scan"
DATASCAN_DISPLAY_NAME = "Vehicles DQ Basic Entity Scan"

logging.basicConfig(level=logging.INFO)
client = dataplex_v1.DataScanServiceClient()
parent_path = f"projects/{PROJECT_ID}/locations/{REGION}"
entity_path = f"{parent_path}/lakes/{LAKE_ID}/zones/{ZONE_ID}/entities/{ENTITY_ID}"
scan_path = f"{parent_path}/dataScans/{DATASCAN_ID}"

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

# --- CREATE DATA SOURCE ---
# Use the correct data source structure for the latest API
data_source = DataSource(
    entity=entity_path  # Reference to the Dataplex entity
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
