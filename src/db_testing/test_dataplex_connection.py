"""
Test script to verify Dataplex API connection and permissions

This script performs basic connectivity tests to ensure:
1. Authentication is working
2. Dataplex API is accessible
3. Required permissions are available
"""

import logging
from google.cloud import dataplex_v1
from google.auth import default
from google.auth.exceptions import DefaultCredentialsError

# Configuration
PROJECT_ID = "dt-denis-sandbox-dev"  # Update with your project ID
REGION = "us-central1"  # Update with your region

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_authentication():
    """Test Google Cloud authentication"""
    try:
        credentials, project = default()
        logger.info(f"✅ Authentication successful")
        logger.info(f"   Default project: {project}")
        return True
    except DefaultCredentialsError as e:
        logger.error(f"❌ Authentication failed: {e}")
        logger.error("   Please ensure you have set up authentication:")
        logger.error("   - Run 'gcloud auth application-default login'")
        logger.error("   - Or set GOOGLE_APPLICATION_CREDENTIALS environment variable")
        return False

def test_dataplex_api():
    """Test Dataplex API connectivity"""
    try:
        client = dataplex_v1.DataScanServiceClient()
        parent_path = f"projects/{PROJECT_ID}/locations/{REGION}"
        
        # Try to list existing data scans (this tests API access and permissions)
        request = dataplex_v1.ListDataScansRequest(parent=parent_path)
        response = client.list_data_scans(request=request)
        
        scan_count = len(list(response))
        logger.info(f"✅ Dataplex API accessible")
        logger.info(f"   Found {scan_count} existing data scans in {REGION}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Dataplex API test failed: {e}")
        logger.error("   Please ensure:")
        logger.error("   - Dataplex API is enabled in your project")
        logger.error("   - You have the required IAM permissions")
        logger.error("   - The project ID and region are correct")
        return False

def test_bigquery_access():
    """Test BigQuery access (needed for data quality scans)"""
    try:
        from google.cloud import bigquery
        client = bigquery.Client(project=PROJECT_ID)
        
        # Test basic BigQuery access
        datasets = list(client.list_datasets())
        logger.info(f"✅ BigQuery access verified")
        logger.info(f"   Found {len(datasets)} datasets in project")
        
        return True
    except Exception as e:
        logger.error(f"❌ BigQuery access test failed: {e}")
        return False

def run_tests():
    """Run all tests"""
    logger.info("🧪 Starting Dataplex setup verification tests...")
    logger.info(f"   Project: {PROJECT_ID}")
    logger.info(f"   Region: {REGION}")
    logger.info("-" * 50)
    
    tests = [
        ("Authentication", test_authentication),
        ("Dataplex API", test_dataplex_api),
        ("BigQuery Access", test_bigquery_access)
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"Running {test_name} test...")
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            logger.error(f"Unexpected error in {test_name} test: {e}")
            results.append(False)
        logger.info("")
    
    # Summary
    logger.info("=" * 50)
    logger.info("TEST SUMMARY")
    logger.info("=" * 50)
    
    for i, (test_name, _) in enumerate(tests):
        status = "✅ PASSED" if results[i] else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
    
    if all(results):
        logger.info("\n🎉 All tests passed! Your setup is ready for data quality scans.")
    else:
        logger.info("\n⚠️  Some tests failed. Please address the issues above before proceeding.")
    
    return all(results)

if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1) 
