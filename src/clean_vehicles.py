#!/usr/bin/env python3
"""
Vehicle Data Cleaning Script

This script processes a large vehicles.csv file by:
1. Loading the data in chunks to handle memory efficiently
2. Removing rows where the 'cylinders' column is null
3. Saving the cleaned data to a configurable output file
4. Uploading the cleaned file to Google Cloud Storage (optional)
5. Loading the cleaned data to BigQuery (optional)

The script is designed to be:
- Parameterized: Accepts command-line arguments and config files
- Repeatable: Can be run multiple times with different configurations
- Idempotent: Overwrites existing outputs consistently

Requirements:
- pandas
- gcsfs
- google-cloud-bigquery (optional, for BigQuery support)
"""

import pandas as pd
import gcsfs
import os
import logging
import argparse
import json
import yaml
import time
from typing import Optional, Dict, Any
from pathlib import Path
import sys
import random

# Optional progress bar support
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

# Retry decorator for network operations
def retry_on_failure(max_attempts: int = 3, base_delay: float = 1.0, max_delay: float = 60.0):
    """Decorator to retry functions on failure with exponential backoff."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:  # Last attempt
                        raise e
                    
                    # Calculate delay with exponential backoff and jitter
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    jitter = random.uniform(0.1, 0.3) * delay
                    total_delay = delay + jitter
                    
                    logger = logging.getLogger(__name__)
                    logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                    logger.info(f"Retrying in {total_delay:.2f} seconds...")
                    time.sleep(total_delay)
            
            return None  # This should never be reached
        return wrapper
    return decorator

# Set up logging
def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """Set up logging configuration."""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=handlers
    )
    return logging.getLogger(__name__)

class VehicleDataCleaner:
    """Vehicle data cleaning and processing class."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize with configuration."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Extract configuration
        self.input_file = config['input_file']
        self.output_file = config['output_file']
        self.chunk_size = config.get('chunk_size', 10000)
        self.filter_column = config.get('filter_column', 'cylinders')
        
        # Cloud storage configuration
        self.gcs_bucket_path = config.get('gcs_bucket_path')
        self.upload_to_gcs = config.get('upload_to_gcs', False)
        
        # BigQuery configuration
        self.bq_dataset = config.get('bq_dataset')
        self.bq_table = config.get('bq_table')
        self.bq_project = config.get('bq_project')
        self.load_to_bigquery = config.get('load_to_bigquery', False)
        
        # Processing options
        self.overwrite_output = config.get('overwrite_output', True)
        self.dry_run = config.get('dry_run', False)
        self.show_progress = config.get('show_progress', True) and TQDM_AVAILABLE
    
    def validate_config(self) -> bool:
        """Validate the configuration."""
        if not os.path.exists(self.input_file):
            self.logger.error(f"Input file {self.input_file} not found")
            return False
        
        if self.upload_to_gcs and not self.gcs_bucket_path:
            self.logger.error("GCS bucket path is required when upload_to_gcs is True")
            return False
        
        if self.load_to_bigquery:
            if not all([self.bq_project, self.bq_dataset, self.bq_table]):
                self.logger.error("BigQuery project, dataset, and table are required when load_to_bigquery is True")
                return False
        
        return True
    
    def estimate_total_rows(self) -> int:
        """Estimate total rows in the input file efficiently."""
        try:
            # Read a sample of rows to estimate file structure
            sample_size = min(1000, self.chunk_size)
            sample_df = pd.read_csv(self.input_file, nrows=sample_size, encoding='utf-8')
            
            # Get file size
            file_size = os.path.getsize(self.input_file)
            
            # Estimate row size (including headers and newlines)
            # Use memory usage as a rough proxy for data size
            sample_memory = sample_df.memory_usage(deep=True).sum()
            estimated_row_size = sample_memory / len(sample_df) if len(sample_df) > 0 else 1000
            
            # Estimate total rows based on file size
            estimated_rows = int(file_size / estimated_row_size * 0.8)  # Conservative estimate
            
            self.logger.info(f"Sample size: {len(sample_df)} rows")
            self.logger.info(f"Estimated average row size: {estimated_row_size:.2f} bytes")
            
            return max(estimated_rows, 1)  # At least 1 row
            
        except Exception as e:
            self.logger.warning(f"Could not estimate row count: {e}")
            # Fallback: use file size / conservative bytes per row
            file_size = os.path.getsize(self.input_file)
            return max(int(file_size / 500), 1)  # Assume ~500 bytes per row as fallback

    def estimate_total_chunks(self) -> int:
        """Estimate total number of chunks for progress tracking."""
        estimated_rows = self.estimate_total_rows()
        estimated_chunks = max(1, int(estimated_rows / self.chunk_size))
        return estimated_chunks

    def check_output_exists(self) -> Dict[str, bool]:
        """Check if output files/objects already exist."""
        exists = {
            'local_file': os.path.exists(self.output_file),
            'gcs_object': False,
            'bq_table': False
        }
        
        # Check GCS object
        if self.upload_to_gcs and self.gcs_bucket_path:
            try:
                fs = gcsfs.GCSFileSystem()
                exists['gcs_object'] = fs.exists(self.gcs_bucket_path)
            except Exception as e:
                self.logger.warning(f"Could not check GCS object existence: {e}")
        
        # Check BigQuery table
        if self.load_to_bigquery:
            try:
                from google.cloud import bigquery
                client = bigquery.Client(project=self.bq_project)
                table_id = f"{self.bq_project}.{self.bq_dataset}.{self.bq_table}"
                try:
                    client.get_table(table_id)
                    exists['bq_table'] = True
                except Exception:
                    exists['bq_table'] = False
            except ImportError:
                self.logger.warning("google-cloud-bigquery not installed, skipping BigQuery table check")
            except Exception as e:
                self.logger.warning(f"Could not check BigQuery table existence: {e}")
        
        return exists
    
    def clean_vehicles_data(self) -> Dict[str, int]:
        """Clean vehicle data by removing rows with null values in the filter column."""
        self.logger.info(f"Starting to process {self.input_file}")
        
        if self.dry_run:
            self.logger.info("DRY RUN MODE - No files will be modified")
        
        # Get file size for progress tracking
        file_size = os.path.getsize(self.input_file)
        self.logger.info(f"Input file size: {file_size / (1024**3):.2f} GB")
        
        # Check if output exists and handle accordingly
        exists = self.check_output_exists()
        if exists['local_file']:
            if self.overwrite_output:
                self.logger.info(f"Output file {self.output_file} exists and will be overwritten")
                if not self.dry_run:
                    os.remove(self.output_file)
            else:
                self.logger.error(f"Output file {self.output_file} exists and overwrite_output is False")
                raise FileExistsError(f"Output file {self.output_file} already exists")
        
        # Initialize counters
        stats = {
            'total_rows': 0,
            'cleaned_rows': 0,
            'filtered_rows': 0
        }
        
        if self.dry_run:
            self.logger.info("Dry run - would process file in chunks...")
            
            # Estimate and log total rows for dry-run
            estimated_rows = self.estimate_total_rows()
            self.logger.info(f"Estimated total rows to process: {estimated_rows:,}")
            
            # Estimate null rows based on sample
            try:
                sample_size = min(1000, self.chunk_size)
                sample_df = pd.read_csv(self.input_file, nrows=sample_size, encoding='utf-8')
                
                if self.filter_column in sample_df.columns:
                    null_ratio = sample_df[self.filter_column].isnull().sum() / len(sample_df)
                    estimated_null_rows = int(estimated_rows * null_ratio)
                    estimated_clean_rows = estimated_rows - estimated_null_rows
                    
                    self.logger.info(f"Estimated rows with null {self.filter_column}: {estimated_null_rows:,} ({null_ratio:.2%})")
                    self.logger.info(f"Estimated final clean rows: {estimated_clean_rows:,}")
                else:
                    self.logger.warning(f"Column '{self.filter_column}' not found in sample data")
                    self.logger.info(f"Available columns: {list(sample_df.columns)}")
                    
            except Exception as e:
                self.logger.warning(f"Could not estimate filtering stats: {e}")
            
            return stats
        
        # Process file in chunks and write to output
        self.logger.info("Processing file in chunks...")
        
        try:
            # Read the file in chunks
            chunk_reader = pd.read_csv(self.input_file, chunksize=self.chunk_size)
            
            # Set up progress bar if enabled
            if self.show_progress:
                # Estimate total chunks for progress bar
                estimated_chunks = self.estimate_total_chunks()
                progress_bar = tqdm(
                    total=estimated_chunks,
                    desc="Cleaning Dataset",
                    unit="chunks",
                    unit_scale=False,
                    bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} chunks [{elapsed}<{remaining}, {rate_fmt}]"
                )
            else:
                progress_bar = None
            
            # Process first chunk to get headers and initialize output file
            first_chunk = True
            
            for chunk_num, chunk in enumerate(chunk_reader, 1):
                # Update progress bar
                if progress_bar:
                    progress_bar.set_postfix({
                        'rows': f"{stats['total_rows']:,}",
                        'filtered': f"{stats['filtered_rows']:,}"
                    })
                    progress_bar.update(1)
                
                # Log every 50 chunks when not using progress bar
                if not self.show_progress and chunk_num % 50 == 0:
                    self.logger.info(f"Processing chunk {chunk_num} ({len(chunk)} rows)")
                
                # Count total rows
                stats['total_rows'] += len(chunk)
                
                # Check if filter column exists
                if self.filter_column not in chunk.columns:
                    if chunk_num == 1:  # Only log this once
                        self.logger.warning(f"'{self.filter_column}' column not found in the data")
                        self.logger.info(f"Available columns: {list(chunk.columns)}")
                    # Continue processing without filtering
                    cleaned_chunk = chunk
                else:
                    # Count null values before cleaning
                    null_count_in_chunk = chunk[self.filter_column].isnull().sum()
                    stats['filtered_rows'] += null_count_in_chunk
                    
                    # Remove rows where filter column is null
                    cleaned_chunk = chunk.dropna(subset=[self.filter_column])
                    
                    # Log details for first few chunks or when not using progress bar
                    if chunk_num <= 3 or (not self.show_progress and chunk_num % 50 == 0):
                        self.logger.info(f"Chunk {chunk_num}: Removed {null_count_in_chunk} rows with null {self.filter_column}")
                
                # Count cleaned rows
                stats['cleaned_rows'] += len(cleaned_chunk)
                
                # Write to output file
                if first_chunk:
                    # Write headers for the first chunk
                    cleaned_chunk.to_csv(self.output_file, index=False, mode='w', encoding='utf-8')
                    first_chunk = False
                else:
                    # Append subsequent chunks without headers
                    cleaned_chunk.to_csv(self.output_file, index=False, mode='a', header=False, encoding='utf-8')
                
                # Memory cleanup
                del chunk, cleaned_chunk
            
            # Close progress bar
            if progress_bar:
                progress_bar.close()
        
        except pd.errors.EmptyDataError:
            self.logger.error("The input file is empty or corrupted")
            raise
        except Exception as e:
            self.logger.error(f"Error processing file: {str(e)}")
            raise
        
        # Log final statistics
        self.logger.info("=" * 60)
        self.logger.info("CLEANING SUMMARY")
        self.logger.info("=" * 60)
        self.logger.info(f"Total rows processed: {stats['total_rows']:,}")
        self.logger.info(f"Rows filtered out: {stats['filtered_rows']:,}")
        self.logger.info(f"Final cleaned rows: {stats['cleaned_rows']:,}")
        if stats['total_rows'] > 0:
            self.logger.info(f"Data retention rate: {(stats['cleaned_rows']/stats['total_rows'])*100:.2f}%")
        
        # Check output file size
        if os.path.exists(self.output_file):
            output_size = os.path.getsize(self.output_file)
            self.logger.info(f"Output file size: {output_size / (1024**3):.2f} GB")
            self.logger.info(f"Size reduction: {((file_size - output_size) / file_size) * 100:.2f}%")
        
        self.logger.info(f"Cleaned data saved to {self.output_file}")
        return stats
    
    @retry_on_failure(max_attempts=3, base_delay=2.0, max_delay=60.0)
    def upload_file_to_gcs(self) -> bool:
        """Upload the cleaned file to Google Cloud Storage."""
        if not self.upload_to_gcs:
            self.logger.info("GCS upload disabled, skipping...")
            return True
        
        self.logger.info("Starting upload to Google Cloud Storage...")
        
        if self.dry_run:
            self.logger.info(f"DRY RUN - Would upload {self.output_file} to {self.gcs_bucket_path}")
            return True
        
        if not os.path.exists(self.output_file):
            raise FileNotFoundError(f"Output file {self.output_file} not found")
        
        try:
            # Initialize GCS filesystem
            fs = gcsfs.GCSFileSystem()
            
            # Check if object exists and handle overwrite
            if fs.exists(self.gcs_bucket_path):
                if self.overwrite_output:
                    self.logger.info(f"GCS object {self.gcs_bucket_path} exists and will be overwritten")
                else:
                    self.logger.error(f"GCS object {self.gcs_bucket_path} exists and overwrite_output is False")
                    return False
            
            # Upload file
            self.logger.info(f"Uploading {self.output_file} to {self.gcs_bucket_path}")
            
            # Copy file to GCS (this will overwrite if exists)
            fs.put(self.output_file, self.gcs_bucket_path)
            
            self.logger.info("Upload completed successfully!")
            
            # Verify upload
            if fs.exists(self.gcs_bucket_path):
                file_info = fs.info(self.gcs_bucket_path)
                self.logger.info(f"Uploaded file size: {file_info['size'] / (1024**3):.2f} GB")
                return True
            else:
                self.logger.warning("Upload verification failed - file not found in GCS")
                return False
                
        except Exception as e:
            self.logger.error(f"Error uploading to GCS: {str(e)}")
            self.logger.error("Make sure you have proper GCS credentials configured")
            raise
    
    @retry_on_failure(max_attempts=3, base_delay=2.0, max_delay=60.0)
    def upload_file_to_bigquery(self) -> bool:
        """Upload the cleaned file to BigQuery."""
        if not self.load_to_bigquery:
            self.logger.info("BigQuery load disabled, skipping...")
            return True
        
        try:
            from google.cloud import bigquery
        except ImportError:
            self.logger.error("google-cloud-bigquery not installed. Install with: pip install google-cloud-bigquery")
            return False
        
        self.logger.info("Starting load to BigQuery...")
        
        if self.dry_run:
            self.logger.info(f"DRY RUN - Would load {self.output_file} to {self.bq_project}.{self.bq_dataset}.{self.bq_table}")
            return True
        
        if not os.path.exists(self.output_file):
            raise FileNotFoundError(f"Output file {self.output_file} not found")
        
        try:
            # Initialize BigQuery client
            client = bigquery.Client(project=self.bq_project)
            table_id = f"{self.bq_project}.{self.bq_dataset}.{self.bq_table}"
            
            # Configure load job
            job_config = bigquery.LoadJobConfig(
                source_format=bigquery.SourceFormat.CSV,
                skip_leading_rows=1,  # Skip header row
                autodetect=True,
                write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE if self.overwrite_output else bigquery.WriteDisposition.WRITE_EMPTY
            )
            
            self.logger.info(f"Loading {self.output_file} to {table_id}")
            
            # Load file to BigQuery
            with open(self.output_file, "rb") as source_file:
                job = client.load_table_from_file(source_file, table_id, job_config=job_config)
            
            # Wait for job to complete
            job.result()
            
            # Get table info
            table = client.get_table(table_id)
            self.logger.info(f"BigQuery load completed successfully!")
            self.logger.info(f"Loaded {table.num_rows} rows to {table_id}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading to BigQuery: {str(e)}")
            raise
    
    def process(self) -> Dict[str, Any]:
        """Main processing method."""
        if not self.validate_config():
            raise ValueError("Configuration validation failed")
        
        # Start timing
        start_time = time.time()
        
        results = {
            'stats': {},
            'gcs_upload': False,
            'bq_load': False
        }
        
        try:
            self.logger.info("Starting vehicle data cleaning process...")
            
            # Step 1: Clean the data
            results['stats'] = self.clean_vehicles_data()
            
            # Step 2: Upload to GCS (if enabled)
            results['gcs_upload'] = self.upload_file_to_gcs()
            
            # Step 3: Upload to BigQuery (if enabled)
            results['bq_load'] = self.upload_file_to_bigquery()
            
            # Calculate and log total elapsed time
            elapsed_time = time.time() - start_time
            self.logger.info(f"Total time taken: {elapsed_time:.2f} seconds")
            self.logger.info("Process completed successfully!")
            
            return results
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            self.logger.error(f"Process failed after {elapsed_time:.2f} seconds: {str(e)}")
            raise

def load_config_file(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML or JSON file."""
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file {config_path} not found")
    
    with open(config_path, 'r') as f:
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            return yaml.safe_load(f)
        elif config_path.suffix.lower() == '.json':
            return json.load(f)
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")

def create_default_config() -> Dict[str, Any]:
    """Create default configuration."""
    return {
        'input_file': 'vehicles.csv',
        'output_file': 'vehicles_cleaned.csv',
        'chunk_size': 10000,
        'filter_column': 'cylinders',
        'gcs_bucket_path': 'gs://dt-denis-sandbox-dev-data/processed/vehicles_cleaned.csv',
        'upload_to_gcs': False,
        'bq_project': None,
        'bq_dataset': None,
        'bq_table': None,
        'load_to_bigquery': False,
        'overwrite_output': True,
        'dry_run': False,
        'show_progress': True,
        'log_level': 'INFO',
        'log_file': None
    }

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Clean vehicle data by removing rows with null cylinders",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input/Output files
    parser.add_argument('--input-file', '-i', default='vehicles.csv',
                        help='Input CSV file path')
    parser.add_argument('--output-file', '-o', default='vehicles_cleaned.csv',
                        help='Output CSV file path')
    parser.add_argument('--config-file', '-c',
                        help='Configuration file (YAML or JSON)')
    
    # Processing options
    parser.add_argument('--chunk-size', type=int, default=10000,
                        help='Number of rows to process at a time')
    parser.add_argument('--filter-column', default='cylinders',
                        help='Column to filter null values from')
    
    # Cloud storage options
    parser.add_argument('--gcs-bucket-path',
                        default='gs://dt-denis-sandbox-dev-data/processed/vehicles_cleaned.csv',
                        help='GCS bucket path for upload')
    parser.add_argument('--upload-to-gcs', action='store_true',
                        help='Upload cleaned file to GCS')
    
    # BigQuery options
    parser.add_argument('--bq-project', help='BigQuery project ID')
    parser.add_argument('--bq-dataset', help='BigQuery dataset ID')
    parser.add_argument('--bq-table', help='BigQuery table ID')
    parser.add_argument('--load-to-bigquery', action='store_true',
                        help='Load cleaned data to BigQuery')
    
    # Control options
    parser.add_argument('--overwrite-output', action='store_true', default=True,
                        help='Overwrite existing output files/objects')
    parser.add_argument('--no-overwrite', dest='overwrite_output', action='store_false',
                        help='Do not overwrite existing output files/objects')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be done without making changes')
    parser.add_argument('--show-progress', action='store_true', default=True,
                        help='Show progress bar during processing')
    parser.add_argument('--no-progress', dest='show_progress', action='store_false',
                        help='Disable progress bar')
    
    # Logging options
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        default='INFO', help='Logging level')
    parser.add_argument('--log-file', help='Log file path')
    
    return parser.parse_args()

def main():
    """Main execution function."""
    args = parse_arguments()
    
    # Setup logging
    logger = setup_logging(args.log_level, args.log_file)
    
    try:
        # Load configuration
        if args.config_file:
            logger.info(f"Loading configuration from {args.config_file}")
            config = load_config_file(args.config_file)
        else:
            config = create_default_config()
        
        # Override config with command line arguments
        config.update({
            'input_file': args.input_file,
            'output_file': args.output_file,
            'chunk_size': args.chunk_size,
            'filter_column': args.filter_column,
            'gcs_bucket_path': args.gcs_bucket_path,
            'upload_to_gcs': args.upload_to_gcs,
            'bq_project': args.bq_project,
            'bq_dataset': args.bq_dataset,
            'bq_table': args.bq_table,
            'load_to_bigquery': args.load_to_bigquery,
            'overwrite_output': args.overwrite_output,
            'dry_run': args.dry_run,
            'show_progress': args.show_progress,
            'log_level': args.log_level,
            'log_file': args.log_file
        })
        
        # Remove None values
        config = {k: v for k, v in config.items() if v is not None}
        
        logger.info("Configuration:")
        for key, value in config.items():
            if 'password' in key.lower() or 'secret' in key.lower():
                logger.info(f"  {key}: [REDACTED]")
            else:
                logger.info(f"  {key}: {value}")
        
        # Warn if progress bar is requested but tqdm is not available
        if config.get('show_progress', True) and not TQDM_AVAILABLE:
            logger.warning("tqdm not available - progress bar disabled. Install with: pip install tqdm")
        
        # Initialize and run cleaner
        cleaner = VehicleDataCleaner(config)
        results = cleaner.process()
        
        logger.info("Final Results:")
        logger.info(f"  Processing stats: {results['stats']}")
        logger.info(f"  GCS upload successful: {results['gcs_upload']}")
        logger.info(f"  BigQuery load successful: {results['bq_load']}")
        
    except Exception as e:
        logger.error(f"Script failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
