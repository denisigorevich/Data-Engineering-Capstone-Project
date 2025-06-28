"""
Capstone Project — Data Quality Management (Scheduled with Export)

This advanced script demonstrates:
1. Scheduled data quality scans (daily at 6 AM UTC)
2. BigQuery result export for analysis and reporting
3. Additional data quality rules beyond basic implementation
4. Comprehensive error handling and logging
5. Production-ready features for enterprise use

Data Quality Rules Applied:
- cylinders is not null (Completeness)
- VIN is exactly 17 characters (Validity)
- VIN is unique across all records (Uniqueness)
- year must be between 1900 and 2025 (Additional Validity)

Results Export:
- Automatically exports all scan results to BigQuery table
- Table: denis_sandbox.vehicles_dq_scheduled_export_results
"""

import logging
from google.cloud import dataplex_v1
from google.cloud.dataplex_v1.types import (
    DataScan, DataQualitySpec, DataQualityRule, DataSource, Trigger,
    CreateDataScanRequest, RunDataScanRequest
)

# --- CONFIGURATION ---
PROJECT_ID = "dt-denis-sandbox-dev"
REGION = "us-central1"
DATASET_ID = "denis_sandbox"
TABLE_ID = "vehicles_cleaned"
DATASCAN_ID = "vehicles-dq-scheduled-export-scan"
DATASCAN_DISPLAY_NAME = "Vehicles DQ Scheduled Export Scan"

# Results export configuration - UNIQUE TABLE NAME for demo clarity
RESULTS_DATASET = "denis_sandbox"
RESULTS_TABLE = "vehicles_dq_scheduled_export_results"

# Schedule configuration (run daily at 6 AM UTC)
SCHEDULE_CRON = "0 6 * * *"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataQualityManager:
    def __init__(self, project_id, region):
        self.project_id = project_id
        self.region = region
        self.client = dataplex_v1.DataScanServiceClient()
        self.parent_path = f"projects/{project_id}/locations/{region}"
    
    def create_data_quality_rules(self):
        """Create comprehensive data quality rules"""
        return [
            # Completeness: cylinders is not null
            DataQualityRule(
                non_null_expectation=DataQualityRule.NonNullExpectation(),
                column="cylinders",
                dimension="COMPLETENESS",
                name="cylinders-not-null",
                description="Ensure cylinders field is not null"
            ),

            # Validity: VIN must be exactly 17 characters
            DataQualityRule(
                row_condition_expectation=DataQualityRule.RowConditionExpectation(
                    sql_expression="LENGTH(VIN) = 17"
                ),
                column="VIN",
                dimension="VALIDITY",
                name="vin-length-check",
                description="VIN must be exactly 17 characters"
            ),

            # Uniqueness: VIN must be unique
            DataQualityRule(
                table_condition_expectation=DataQualityRule.TableConditionExpectation(
                    sql_expression="COUNT(DISTINCT VIN) = COUNT(*)"
                ),
                dimension="UNIQUENESS",
                name="vin-uniqueness",
                description="VIN values must be unique across all records"
            ),

            # Additional validity check: year must be reasonable
            DataQualityRule(
                range_expectation=DataQualityRule.RangeExpectation(
                    min_value="1900",
                    max_value="2025"
                ),
                column="year",
                dimension="VALIDITY",
                name="year-range-check",
                description="Year must be between 1900 and 2025"
            )
        ]
    
    def create_post_scan_actions(self):
        """Configure post-scan actions"""
        return DataQualitySpec.PostScanActions(
            bigquery_export=DataQualitySpec.PostScanActions.BigQueryExport(
                results_table=f"//bigquery.googleapis.com/projects/{self.project_id}/datasets/{RESULTS_DATASET}/tables/{RESULTS_TABLE}"
            )
        )
    
    def create_data_scan_spec(self, execution_mode="scheduled", export_results=True):
        """Create the complete DataScan specification"""
        
        # Data quality rules
        dq_rules = self.create_data_quality_rules()
        
        # Data quality spec with optional post-scan actions
        dq_spec_kwargs = {"rules": dq_rules}
        if export_results:
            dq_spec_kwargs["post_scan_actions"] = self.create_post_scan_actions()
        
        dq_spec = DataQualitySpec(**dq_spec_kwargs)

        # Data source (BigQuery table)
        data_source = DataSource(
            resource=f"//bigquery.googleapis.com/projects/{self.project_id}/datasets/{DATASET_ID}/tables/{TABLE_ID}"
        )

        # Execution specification
        if execution_mode == "scheduled":
            trigger = Trigger(
                schedule=Trigger.Schedule(cron=SCHEDULE_CRON)
            )
        else:  # on-demand
            trigger = Trigger(
                on_demand=Trigger.OnDemand()
            )

        execution_spec = DataScan.ExecutionSpec(trigger=trigger)

        # Complete DataScan object
        return DataScan(
            data=data_source,
            data_quality_spec=dq_spec,
            display_name=DATASCAN_DISPLAY_NAME,
            execution_spec=execution_spec,
            description=f"Scheduled data quality scan with export to {RESULTS_DATASET}.{RESULTS_TABLE}"
        )
    
    def create_or_update_scan(self, scan_id, execution_mode="scheduled", export_results=True):
        """Create or update a data quality scan"""
        scan_path = f"{self.parent_path}/dataScans/{scan_id}"
        data_scan = self.create_data_scan_spec(execution_mode, export_results)
        
        try:
            # Try to get existing scan
            existing = self.client.get_data_scan(name=scan_path)
            logger.info(f"DataScan already exists: {existing.name}")
            logger.info("Skipping creation since scan already exists. Delete manually if you want to recreate.")
            return existing

        except Exception as e:
            logger.info(f"DataScan doesn't exist, creating new one. Error: {e}")
            
            # Create new scan
            create_request = CreateDataScanRequest(
                parent=self.parent_path,
                data_scan_id=scan_id,
                data_scan=data_scan
            )
            operation = self.client.create_data_scan(request=create_request)
            result = operation.result()
            logger.info(f"Successfully created new DataScan: {result.name}")
            return result
    
    def run_scan(self, scan_id):
        """Run a data quality scan on demand"""
        scan_path = f"{self.parent_path}/dataScans/{scan_id}"
        
        try:
            run_request = RunDataScanRequest(name=scan_path)
            run_response = self.client.run_data_scan(request=run_request)
            logger.info(f"Successfully triggered scan job: {run_response.job.name}")
            return run_response.job
        except Exception as e:
            logger.error(f"Failed to run scan: {e}")
            raise
    
    def get_scan_results(self, scan_id, job_id=None):
        """Get the latest scan results"""
        scan_path = f"{self.parent_path}/dataScans/{scan_id}"
        
        try:
            if job_id:
                job_path = f"{scan_path}/jobs/{job_id}"
                job = self.client.get_data_scan_job(name=job_path)
            else:
                # Get the most recent scan with results
                scan = self.client.get_data_scan(name=scan_path, view="FULL")
                if hasattr(scan, 'data_quality_result'):
                    return scan.data_quality_result
                else:
                    logger.warning("No scan results available yet")
                    return None
            
            return job.data_quality_result if hasattr(job, 'data_quality_result') else None
        except Exception as e:
            logger.error(f"Failed to get scan results: {e}")
            return None

def main():
    """Main execution function"""
    manager = DataQualityManager(PROJECT_ID, REGION)
    
    try:
        # Create/update scheduled scan with result export
        logger.info("Creating/updating scheduled data quality scan with BigQuery export...")
        scan = manager.create_or_update_scan(
            scan_id=DATASCAN_ID,
            execution_mode="scheduled",
            export_results=True
        )
        
        # Note: Scheduled scans cannot be manually triggered
        logger.info("⚠️  Note: Scheduled scans cannot be manually triggered via API")
        logger.info("   The scan will run automatically according to its schedule")
        
        logger.info("✅ Data quality scan setup complete!")
        logger.info(f"📊 Scan name: {scan.name}")
        logger.info(f"⏰ Schedule: {SCHEDULE_CRON} (daily at 6 AM UTC)")
        logger.info(f"📁 Results will be exported to: {RESULTS_DATASET}.{RESULTS_TABLE}")
        logger.info("🔍 View scan status in Dataplex console")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main() 
