# Vehicle Data Cleaning Script

A parameterized, configurable, and idempotent Python script for cleaning large vehicle CSV files by removing rows with null values in specified columns. The script supports local processing, Google Cloud Storage upload, and BigQuery loading.

## Features

- **Memory Efficient**: Processes large files (1.4GB+) using pandas chunking
- **Parameterized**: Fully configurable via command-line arguments or config files
- **Idempotent**: Can be run multiple times safely, overwrites existing outputs
- **Repeatable**: Consistent results across multiple runs
- **Multi-Destination**: Supports local files, Google Cloud Storage, and BigQuery
- **Dry Run Mode**: Preview operations without making changes
- **Comprehensive Logging**: Detailed progress tracking and statistics

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up Google Cloud credentials (if using GCS or BigQuery):
```bash
# Option 1: Application Default Credentials
gcloud auth application-default login

# Option 2: Service Account Key
export GOOGLE_APPLICATION_CREDENTIALS="path/to/service-account-key.json"
```

## Usage

### Command Line Interface

#### Basic Usage (Local Processing Only)
```bash
python clean_vehicles.py
```

#### Upload to Google Cloud Storage
```bash
python clean_vehicles.py --upload-to-gcs
```

#### Load to BigQuery
```bash
python clean_vehicles.py \
    --load-to-bigquery \
    --bq-project your-project-id \
    --bq-dataset your_dataset \
    --bq-table vehicles_cleaned
```

#### Full Pipeline (Local + GCS + BigQuery)
```bash
python clean_vehicles.py \
    --upload-to-gcs \
    --load-to-bigquery \
    --bq-project your-project-id \
    --bq-dataset vehicle_data \
    --bq-table cleaned_vehicles
```

#### Custom Input/Output Files
```bash
python clean_vehicles.py \
    --input-file data/raw_vehicles.csv \
    --output-file data/processed/clean_vehicles.csv
```

#### Different Filter Column
```bash
python clean_vehicles.py --filter-column engine_size
```

#### Dry Run (Preview Mode)
```bash
python clean_vehicles.py --dry-run --upload-to-gcs --load-to-bigquery
```

### Configuration File Usage

Use a YAML configuration file for complex setups:

```bash
python clean_vehicles.py --config-file config.yaml
```

Example config.yaml:
```yaml
input_file: vehicles.csv
output_file: vehicles_cleaned.csv
chunk_size: 10000
filter_column: cylinders
gcs_bucket_path: gs://your-bucket/processed/vehicles_cleaned.csv
upload_to_gcs: true
bq_project: your-project-id
bq_dataset: vehicle_data
bq_table: cleaned_vehicles
load_to_bigquery: true
overwrite_output: true
dry_run: false
log_level: INFO
```

### Command Line Arguments

| Argument | Short | Default | Description |
|----------|-------|---------|-------------|
| `--input-file` | `-i` | `vehicles.csv` | Input CSV file path |
| `--output-file` | `-o` | `vehicles_cleaned.csv` | Output CSV file path |
| `--config-file` | `-c` | None | Configuration file (YAML/JSON) |
| `--chunk-size` | | `10000` | Rows to process at a time |
| `--filter-column` | | `cylinders` | Column to filter null values |
| `--gcs-bucket-path` | | `gs://dt-denis-sandbox...` | GCS destination path |
| `--upload-to-gcs` | | `False` | Enable GCS upload |
| `--bq-project` | | None | BigQuery project ID |
| `--bq-dataset` | | None | BigQuery dataset ID |
| `--bq-table` | | None | BigQuery table ID |
| `--load-to-bigquery` | | `False` | Enable BigQuery loading |
| `--overwrite-output` | | `True` | Overwrite existing outputs |
| `--no-overwrite` | | | Disable output overwriting |
| `--dry-run` | | `False` | Preview mode (no changes) |
| `--log-level` | | `INFO` | Logging level |
| `--log-file` | | None | Log file path |

## Idempotent Behavior

The script is designed to be idempotent, meaning you can run it multiple times with the same configuration and get consistent results:

- **Local Files**: Existing output files are overwritten (when `overwrite_output=True`)
- **GCS Objects**: Existing objects are replaced with new uploads
- **BigQuery Tables**: Tables are truncated and reloaded with fresh data
- **Error Handling**: Graceful handling of existing resources

## Examples

### Example 1: Basic Local Processing
```bash
python clean_vehicles.py --input-file vehicles.csv --output-file clean_vehicles.csv
```

### Example 2: Different Chunk Size for Memory Optimization
```bash
python clean_vehicles.py --chunk-size 5000  # Smaller chunks for limited memory
```

### Example 3: Complete Pipeline with Logging
```bash
python clean_vehicles.py \
    --upload-to-gcs \
    --load-to-bigquery \
    --bq-project my-project \
    --bq-dataset raw_data \
    --bq-table vehicles \
    --log-level DEBUG \
    --log-file processing.log
```

### Example 4: Preview Before Running
```bash
# First, see what would happen
python clean_vehicles.py --dry-run --upload-to-gcs --load-to-bigquery

# Then run for real
python clean_vehicles.py --upload-to-gcs --load-to-bigquery
```

## Output

The script provides comprehensive logging including:

- File size information
- Processing progress by chunk
- Statistics on rows processed/filtered/retained
- Data retention rates
- Upload/load confirmations
- Error handling and debugging information

Example output:
```
2024-01-15 10:30:00 - INFO - Starting vehicle data cleaning process...
2024-01-15 10:30:01 - INFO - Input file size: 1.42 GB
2024-01-15 10:30:05 - INFO - Processing chunk 1 (10000 rows)
2024-01-15 10:30:05 - INFO - Chunk 1: Removed 245 rows with null cylinders
...
2024-01-15 10:45:00 - INFO - ============================================================
2024-01-15 10:45:00 - INFO - CLEANING SUMMARY
2024-01-15 10:45:00 - INFO - ============================================================
2024-01-15 10:45:00 - INFO - Total rows processed: 1,248,932
2024-01-15 10:45:00 - INFO - Rows filtered out: 12,847
2024-01-15 10:45:00 - INFO - Final cleaned rows: 1,236,085
2024-01-15 10:45:00 - INFO - Data retention rate: 99.03%
2024-01-15 10:45:00 - INFO - Output file size: 1.38 GB
```

## Error Handling

The script includes robust error handling for:
- Missing input files
- Invalid configurations
- GCS authentication issues
- BigQuery connection problems
- Memory constraints
- File permission issues

## Requirements

- Python 3.7+
- pandas >= 1.3.0
- gcsfs >= 2021.0.0
- pyyaml >= 5.4.0
- google-cloud-bigquery >= 3.0.0 (optional, for BigQuery features)

## License

This script is provided as-is for data processing tasks. 
