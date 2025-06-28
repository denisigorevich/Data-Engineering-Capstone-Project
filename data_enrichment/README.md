# Vehicle Data Enrichment Pipeline

This directory contains a comprehensive data enrichment pipeline that leverages the NHTSA Vehicle API to enrich vehicle datasets with missing cylinder count information and additional vehicle metadata.

## 🚀 Features

### Core Capabilities
- **Robust API Integration**: Seamless integration with NHTSA Vehicle API
- **Intelligent Rate Limiting**: Respects API constraints with configurable request rates
- **Batch Processing**: Efficient processing of large datasets with memory optimization
- **Error Handling & Retry Logic**: Exponential backoff and comprehensive error recovery
- **Data Validation**: VIN format validation and data quality checks
- **Multiple Output Formats**: Support for CSV, Parquet, and BigQuery

### Advanced Features
- **Resume Capability**: Checkpoint-based processing for large datasets
- **Threading Support**: Concurrent API calls for improved performance
- **Progress Tracking**: Real-time progress bars and detailed logging
- **Configuration-Driven**: YAML-based configuration for easy customization
- **Comprehensive Logging**: Structured logging with configurable levels

## 📁 Files Overview

| File | Purpose |
|------|---------|
| `enrich_with_cylinders.py` | Main enrichment script with full pipeline functionality |
| `batch_enrich.py` | Advanced batch processing with checkpointing for large datasets |
| `test_api.py` | API testing and validation script |
| `config.yaml` | Configuration file with sensible defaults |
| `README.md` | This documentation file |

## 🛠 Installation

### Prerequisites
Ensure you have Python 3.8+ and install the required dependencies:

```bash
pip install -r ../requirements.txt
```

### Key Dependencies
- `requests>=2.28.0` - HTTP requests for API calls
- `pandas>=1.3.0` - Data manipulation and analysis
- `pyyaml>=5.4.0` - Configuration file parsing
- `tqdm>=4.60.0` - Progress bars (optional)
- `pyarrow>=10.0.0` - Parquet support (optional)

## 🚀 Quick Start

### 1. Test API Connectivity
First, verify that the NHTSA API is accessible:

```bash
# Test with sample VINs
python test_api.py

# Test with a specific VIN
python test_api.py --vin 3GTP1VEC4EG551563

# Test with real data from your dataset
python test_api.py --data-file ../data/vehicles_cleaned.csv --sample-size 5
```

### 2. Basic Enrichment
Enrich your vehicle dataset with cylinder information:

```bash
# Basic usage with default settings
python enrich_with_cylinders.py \
    --input ../data/vehicles_cleaned.csv \
    --output ../data/vehicles_enriched.csv

# With custom configuration
python enrich_with_cylinders.py --config config.yaml
```

### 3. Large Dataset Processing
For datasets larger than 1GB, use the batch processor:

```bash
# Process large file with checkpointing
python batch_enrich.py \
    --input ../data/vehicles_cleaned.csv \
    --output ../data/vehicles_enriched.csv \
    --chunk-size 5000 \
    --save-frequency 10

# Resume interrupted processing
python batch_enrich.py \
    --input ../data/vehicles_cleaned.csv \
    --output ../data/vehicles_enriched.csv \
    --resume
```

## ⚙️ Configuration

### Configuration File (config.yaml)
```yaml
# API Configuration
api:
  max_requests_per_second: 5.0  # Respect API rate limits
  timeout: 30                   # Request timeout in seconds
  max_retries: 3               # Retry attempts for failed requests

# Processing Configuration
processing:
  batch_size: 100              # VINs per batch
  use_threading: true          # Enable concurrent processing
  max_workers: 5               # Concurrent thread limit
  skip_existing_cylinders: true # Skip rows with existing data

# Output Configuration
output:
  format: "csv"                # csv or parquet
  include_api_metadata: true   # Include additional API fields

# Logging Configuration
logging:
  level: "INFO"                # DEBUG, INFO, WARNING, ERROR
  file: null                   # Optional log file path
```

### Command Line Options

#### enrich_with_cylinders.py
```bash
python enrich_with_cylinders.py [OPTIONS]

Options:
  -i, --input PATH              Input vehicle data file (CSV/Parquet)
  -o, --output PATH             Output enriched data file
  -c, --config PATH             Configuration file (YAML)
  --batch-size INTEGER          Batch size for processing (default: 100)
  --max-workers INTEGER         Maximum worker threads (default: 5)
  --rate-limit FLOAT            Max requests per second (default: 5.0)
  --log-level [DEBUG|INFO|WARNING|ERROR]  Logging level
  --log-file PATH               Log file path
  --no-threading                Disable threading
  --include-all-cylinders       Enrich all VINs (ignore existing data)
```

#### batch_enrich.py
```bash
python batch_enrich.py [OPTIONS]

Options:
  -i, --input PATH              Input vehicle data file (CSV)
  -o, --output PATH             Output enriched data file
  -c, --config PATH             Configuration file (YAML)
  --chunk-size INTEGER          Rows per chunk (default: 10000)
  --save-frequency INTEGER      Checkpoint save frequency (default: 5)
  --checkpoint-dir PATH         Checkpoint directory (default: checkpoints)
  --resume                      Resume from checkpoint
  --log-level [DEBUG|INFO|WARNING|ERROR]  Logging level
```

## 📊 Data Schema

### Input Requirements
Your input data must contain a `VIN` column with 17-character vehicle identification numbers.

### Output Schema
The enriched dataset includes the original columns plus:

| Column | Type | Description |
|--------|------|-------------|
| `api_cylinders` | Integer | Number of cylinders from NHTSA API |
| `api_make` | String | Vehicle manufacturer |
| `api_model` | String | Vehicle model |
| `api_year` | String | Model year |
| `api_engine_displacement` | String | Engine displacement in liters |
| `api_fuel_type` | String | Primary fuel type |
| `api_response_time` | Float | API response time in seconds |
| `api_error_message` | String | Error message if enrichment failed |

The original `cylinders` column is updated with API data where the original value was missing.

## 🔧 Advanced Usage

### Custom Configuration
Create a custom configuration file for your specific needs:

```yaml
# High-throughput configuration
api:
  max_requests_per_second: 10.0
  timeout: 15
  max_retries: 5

processing:
  batch_size: 200
  use_threading: true
  max_workers: 10
  skip_existing_cylinders: false  # Re-enrich all VINs

output:
  format: "parquet"
  include_api_metadata: false     # Minimal output

logging:
  level: "WARNING"
  file: "enrichment.log"
```

### Memory-Efficient Processing
For very large datasets (>10GB), use chunked processing:

```bash
# Process 50GB file in 5K row chunks
python batch_enrich.py \
    --input huge_dataset.csv \
    --output enriched_huge_dataset.csv \
    --chunk-size 5000 \
    --save-frequency 20 \
    --log-level WARNING
```

### Integration with BigQuery
For cloud-scale processing, consider uploading results to BigQuery:

```python
from google.cloud import bigquery

# Load enriched data to BigQuery
client = bigquery.Client()
job_config = bigquery.LoadJobConfig(
    source_format=bigquery.SourceFormat.CSV,
    skip_leading_rows=1,
    autodetect=True,
)

with open("vehicles_enriched.csv", "rb") as source_file:
    job = client.load_table_from_file(
        source_file,
        "your-project.vehicle_data.enriched_vehicles",
        job_config=job_config,
    )

job.result()  # Wait for the job to complete
```

## 🚦 Performance Guidelines

### Rate Limiting
The NHTSA API doesn't specify official rate limits, but we recommend:
- **Development/Testing**: 1-2 requests/second
- **Production**: 5-10 requests/second
- **Bulk Processing**: Monitor response times and adjust accordingly

### Optimal Settings by Dataset Size

| Dataset Size | Chunk Size | Batch Size | Workers | Save Frequency |
|--------------|------------|------------|---------|----------------|
| < 100K rows  | N/A        | 100        | 5       | N/A            |
| 100K - 1M    | 10,000     | 100        | 5       | 5              |
| 1M - 10M     | 10,000     | 50         | 3       | 10             |
| > 10M        | 5,000      | 25         | 2       | 20             |

### Memory Considerations
- Each worker thread uses ~50-100MB of memory
- Chunk size affects peak memory usage
- For memory-constrained environments, use smaller chunks and fewer workers

## 🛡️ Error Handling

### Common Issues and Solutions

1. **API Timeout Errors**
   ```
   Solution: Increase timeout in config or reduce request rate
   ```

2. **Invalid VIN Format**
   ```
   Solution: Check VIN validation - must be 17 alphanumeric characters (no I, O, Q)
   ```

3. **Memory Issues with Large Files**
   ```
   Solution: Use batch_enrich.py with smaller chunk sizes
   ```

4. **Network Connectivity Issues**
   ```
   Solution: Enable retry logic and use resume capability
   ```

### Retry Logic
The pipeline implements exponential backoff with jitter:
- Initial delay: 1 second
- Maximum delay: 30 seconds
- Maximum attempts: 3 (configurable)

## 📈 Monitoring and Logging

### Log Levels
- **DEBUG**: Detailed API request/response information
- **INFO**: Progress updates and summary statistics
- **WARNING**: Recoverable errors and rate limiting
- **ERROR**: Fatal errors requiring intervention

### Key Metrics to Monitor
- Success rate (aim for >95%)
- Average response time (should be <2 seconds)
- Error distribution by type
- Processing throughput (VINs/minute)

### Sample Log Output
```
2024-06-28 01:15:23 - INFO - Starting vehicle data enrichment pipeline
2024-06-28 01:15:23 - INFO - Loaded 100,000 rows from data/vehicles_cleaned.csv
2024-06-28 01:15:23 - INFO - Total VINs to enrich: 85,432 (from 100,000 total rows)
2024-06-28 01:15:24 - INFO - Processing batch 1/855 (100 VINs)
2024-06-28 01:15:26 - INFO - Batch 1 complete: 98 successful, 2 failed, avg response time: 1.85s
```

## 🤝 Contributing

### Adding New Features
1. Follow the existing code structure and patterns
2. Add comprehensive logging and error handling
3. Include unit tests for new functionality
4. Update documentation

### API Extensions
To add support for additional NHTSA API fields:
1. Update the `EnrichmentResult` dataclass
2. Modify the `get_vehicle_info` method to extract new fields
3. Update the configuration schema
4. Add field mapping in the merge logic

## 📚 API Reference

### NHTSA Vehicle API
- **Base URL**: `https://vpic.nhtsa.dot.gov/api/vehicles/decodevinvalues/`
- **Format**: `{BASE_URL}/{VIN}?format=json`
- **Documentation**: [NHTSA vPIC API](https://vpic.nhtsa.dot.gov/api/)

### Available Fields
The API provides 100+ vehicle attributes. This pipeline focuses on:
- EngineCylinders
- Make
- Model
- ModelYear
- DisplacementL
- FuelTypePrimary

## 🔍 Troubleshooting

### Performance Issues
```bash
# Check API connectivity
python test_api.py --vin 3GTP1VEC4EG551563

# Test with minimal configuration
python enrich_with_cylinders.py \
    --input small_sample.csv \
    --output test_output.csv \
    --batch-size 10 \
    --max-workers 1 \
    --rate-limit 1.0
```

### Data Quality Issues
```bash
# Validate VIN format in your dataset
python -c "
import pandas as pd
from enrich_with_cylinders import validate_vin

df = pd.read_csv('your_data.csv')
invalid_vins = df[~df['VIN'].apply(validate_vin)]
print(f'Invalid VINs: {len(invalid_vins)}')
print(invalid_vins['VIN'].head())
"
```

### Resume Interrupted Processing
```bash
# Check available checkpoints
ls checkpoints/

# Resume specific processing run
python batch_enrich.py \
    --input ../data/vehicles_cleaned.csv \
    --output ../data/vehicles_enriched.csv \
    --resume
```

## 📄 License

This data enrichment pipeline is part of the capstone project and follows the same licensing terms as the parent project.

## 🆘 Support

For issues, questions, or contributions:
1. Check the troubleshooting section above
2. Review log files for detailed error information
3. Test with a small sample dataset first
4. Ensure all dependencies are properly installed

---

**Happy Enriching! 🚗💨** 
