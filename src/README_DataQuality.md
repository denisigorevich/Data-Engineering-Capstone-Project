# Dataplex Data Quality Management Scripts

This directory contains updated Python scripts for managing data quality scans using Google Cloud Dataplex. These scripts are compatible with the latest Dataplex Python client API (version 2.10.0+).

## Files Overview

### Core Scripts

1. **`dq_basic_dataplex_entity.py`** - Basic scan using Dataplex entities (lakes/zones)
2. **`dq_simple_bigquery_ondemand.py`** - Simple BigQuery table scan (on-demand only)
3. **`dq_scheduled_with_export.py`** - Advanced scheduled scan with BigQuery export
4. **`test_dataplex_connection.py`** - Connection and permission verification

### Script Comparison for Demo

| Feature | Basic Entity | Simple OnDemand | Scheduled Export |
|---------|--------------|------------------|------------------|
| **Data Source** | Dataplex Entity | BigQuery Table | BigQuery Table |
| **Execution** | On-demand only | On-demand only | **Scheduled** + On-demand |
| **Scan ID** | `vehicles-dq-basic-entity-scan` | `vehicles-dq-simple-ondemand-scan` | `vehicles-dq-scheduled-export-scan` |
| **Result Export** | ❌ Console only | ❌ Console only | ✅ **BigQuery Export** |
| **Export Table** | None | None | `denis_sandbox.vehicles_dq_scheduled_export_results` |
| **Rules Count** | 3 basic rules | 3 basic rules | **4 rules** (includes year validation) |
| **Best For** | Learning Dataplex entities | Quick testing | **Production use** |

### Key Demo Points

The **file names clearly indicate their purpose**:

- **Basic**: Shows entity-based scanning (requires Dataplex setup)
- **Simple**: Shows direct BigQuery scanning (easiest to start)
- **Scheduled**: Shows enterprise features (automation + reporting)

The **result tables are uniquely named** to avoid confusion:
- Previous table: `vehicles_quality_scan_results` (from old script)
- New table: `vehicles_dq_scheduled_export_results` (from scheduled script)

### Key Fixes Applied

The original script had several API compatibility issues that have been resolved:

- ✅ **Data Source Structure**: Changed from `data_source=entity_path` to `data=DataSource(...)`
- ✅ **Execution Specification**: Updated trigger and scheduling structure
- ✅ **API Calls**: Fixed request object creation and method calls
- ✅ **Error Handling**: Added proper exception handling and logging
- ✅ **BigQuery Integration**: Added direct BigQuery table support
- ✅ **Unique Naming**: Each script creates distinctly named scans and tables

## Prerequisites

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Authentication

Set up Google Cloud authentication:

```bash
gcloud auth application-default login
```

### 3. Enable APIs

Ensure these APIs are enabled in your Google Cloud project:

```bash
gcloud services enable dataplex.googleapis.com
gcloud services enable bigquery.googleapis.com
```

### 4. IAM Permissions

Your account needs these IAM roles:
- `roles/dataplex.dataScanAdmin` (to create/manage data scans)
- `roles/bigquery.dataViewer` (to read BigQuery tables)
- `roles/bigquery.dataEditor` (if exporting results to BigQuery)

## Usage Guide

### Step 1: Test Your Setup

Before running data quality scans, verify your setup:

```bash
python test_dataplex_connection.py
```

This will test authentication, API access, and permissions.

### Step 2: Choose Your Script for Demo

#### Option A: Simple BigQuery Scan (Recommended for Starting)

Use `dq_simple_bigquery_ondemand.py` for the simplest demonstration:

```bash
python dq_simple_bigquery_ondemand.py
```

**Demo Points:**
- Fastest to run and understand
- No Dataplex setup required
- Results visible in Dataplex console only

#### Option B: Basic Dataplex Entity Scan

Use `dq_basic_dataplex_entity.py` if you have Dataplex lakes/zones set up:

```bash
python dq_basic_dataplex_entity.py
```

**Demo Points:**
- Shows integration with Dataplex data management
- Requires existing lake/zone/entity setup
- Good for showing enterprise data governance

#### Option C: Scheduled with Export (Best for Demo)

Use `dq_scheduled_with_export.py` for the most impressive demonstration:

```bash
python dq_scheduled_with_export.py
```

**Demo Points:**
- Shows automated scheduling (daily at 6 AM)
- Exports results to `denis_sandbox.vehicles_dq_scheduled_export_results`
- Includes additional year validation rule
- Production-ready features

## Data Quality Rules

### Basic Rules (All Scripts)
1. **Completeness**: `cylinders` column must not be null
2. **Validity**: `VIN` must be exactly 17 characters  
3. **Uniqueness**: `VIN` values must be unique

### Additional Rules (Scheduled Export Only)
4. **Validity**: `year` must be between 1900 and 2025

## Configuration for Your Environment

Update these values in each script:

```python
PROJECT_ID = "your-project-id"
DATASET_ID = "your-dataset"
TABLE_ID = "your-table"
```

## Demo Flow Recommendation

For best demo impact, run scripts in this order:

1. **Test Connection**: `python test_dataplex_connection.py`
2. **Simple Demo**: `python dq_simple_bigquery_ondemand.py`
3. **Advanced Demo**: `python dq_scheduled_with_export.py`
4. **Show Results**: Query `denis_sandbox.vehicles_dq_scheduled_export_results` in BigQuery

## Troubleshooting

### Common Issues

1. **Scan Already Exists**
   ```
   google.api_core.exceptions.AlreadyExists: 409 Resource already exists
   ```
   **Solution**: Delete the existing scan in Dataplex console or use unique DATASCAN_ID

2. **Authentication Errors**
   ```
   google.auth.exceptions.DefaultCredentialsError
   ```
   **Solution**: Run `gcloud auth application-default login`

3. **Table Not Found**
   ```
   google.api_core.exceptions.NotFound: 404 Table not found
   ```
   **Solution**: Verify your PROJECT_ID, DATASET_ID, and TABLE_ID are correct

### Debug Mode

Enable debug logging in any script:

```python
logging.basicConfig(level=logging.DEBUG)
```

## Next Steps

After your demo:

1. **Monitor Results**: Check scan results in the Dataplex console
2. **Analyze Data**: Query the exported results in BigQuery
3. **Set Up Alerts**: Configure notifications for data quality failures
4. **Scale**: Apply similar scans to other datasets in your organization

## API Documentation

For more details on the Dataplex API:
- [Dataplex Python Client Documentation](https://cloud.google.com/python/docs/reference/dataplex/latest)
- [Dataplex Data Quality API Guide](https://cloud.google.com/dataplex/docs/auto-data-quality-overview)
- [BigQuery Integration](https://cloud.google.com/dataplex/docs/use-auto-data-quality)

## Support

If you encounter issues:
1. Run the test script to verify your setup
2. Check the Google Cloud Console for detailed error messages
3. Review the Dataplex documentation for API changes
4. Ensure you're using the latest version of `google-cloud-dataplex` 
