#!/usr/bin/env python3
"""
Vehicle Data Enrichment Script using NHTSA Vehicle API

This script enriches vehicle data by querying the NHTSA Vehicle API to retrieve
missing cylinder count information for vehicles based on their VIN numbers.

Features:
- Robust error handling and retry logic with exponential backoff
- Rate limiting to respect API constraints
- Batch processing for large datasets
- Comprehensive logging and progress tracking
- Data validation and quality checks
- Configuration-driven approach
- Support for multiple output formats (CSV, Parquet, BigQuery)

Usage:
    python enrich_with_cylinders.py --input data/vehicles_cleaned.csv --output data/vehicles_enriched.csv
    python enrich_with_cylinders.py --config config/enrichment_config.yaml
"""

import requests
import pandas as pd
import time
import logging
import argparse
import yaml
import json
import random
from typing import Optional, Dict, Any, List
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from urllib.parse import quote
import re

# Optional progress bar support
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

# Optional pyarrow support for Parquet
try:
    import importlib.util
    PYARROW_AVAILABLE = importlib.util.find_spec("pyarrow") is not None
except ImportError:
    PYARROW_AVAILABLE = False


@dataclass
class EnrichmentResult:
    """Data class for enrichment API results."""
    vin: str
    cylinders: Optional[int]
    api_response_time: float
    error_message: Optional[str] = None
    make: Optional[str] = None
    model: Optional[str] = None
    year: Optional[str] = None
    engine_displacement: Optional[str] = None
    fuel_type: Optional[str] = None


class RateLimiter:
    """Simple rate limiter to control API request frequency."""
    
    def __init__(self, max_requests_per_second: float = 5.0):
        self.max_requests_per_second = max_requests_per_second
        self.min_interval = 1.0 / max_requests_per_second
        self.last_request_time = 0.0
    
    def wait_if_needed(self):
        """Wait if necessary to maintain rate limit."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_interval:
            sleep_time = self.min_interval - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()


def retry_with_backoff(max_attempts: int = 3, base_delay: float = 1.0, max_delay: float = 60.0):
    """Decorator for retry logic with exponential backoff."""
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
                    jitter = random.uniform(0.1, 0.3) * delay  # nosec B311
                    total_delay = delay + jitter
                    
                    logger = logging.getLogger(__name__)
                    logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                    logger.info(f"Retrying in {total_delay:.2f} seconds...")
                    time.sleep(total_delay)
            
            return None  # This should never be reached
        return wrapper
    return decorator


def validate_vin(vin: str) -> bool:
    """
    Validate VIN format (17 characters, alphanumeric, no I, O, Q).
    
    Args:
        vin (str): The VIN to validate
    
    Returns:
        bool: True if VIN is valid format
    """
    if not vin or not isinstance(vin, str):
        return False
    
    # Remove whitespace and convert to uppercase
    vin = vin.strip().upper()
    
    # Check length
    if len(vin) != 17:
        return False
    
    # Check for invalid characters (I, O, Q are not allowed in VINs)
    if re.search(r'[IOQ]', vin):
        return False
    
    # Check that it's alphanumeric
    if not vin.isalnum():
        return False
    
    return True


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None, structured: bool = False) -> logging.Logger:
    """Set up logging configuration with optional structured JSON format."""
    if structured:
        # Structured JSON logging for production
        import json
        
        class JSONFormatter(logging.Formatter):
            def format(self, record):
                log_data = {
                    'timestamp': self.formatTime(record, self.datefmt),
                    'level': record.levelname,
                    'logger': record.name,
                    'message': record.getMessage(),
                    'module': record.module,
                    'function': record.funcName,
                    'line': record.lineno
                }
                
                # Add extra fields if present
                if hasattr(record, 'vin'):
                    log_data['vin'] = record.vin
                if hasattr(record, 'response_time'):
                    log_data['response_time'] = record.response_time
                if hasattr(record, 'batch_id'):
                    log_data['batch_id'] = record.batch_id
                if hasattr(record, 'success_count'):
                    log_data['success_count'] = record.success_count
                if hasattr(record, 'error_count'):
                    log_data['error_count'] = record.error_count
                
                return json.dumps(log_data)
        
        formatter = JSONFormatter()
    else:
        # Standard logging format
        log_format = '%(asctime)s %(levelname)s [%(name)s:%(funcName)s:%(lineno)d] %(message)s'
        formatter = logging.Formatter(log_format, datefmt='%Y-%m-%d %H:%M:%S')
    
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    # Configure handlers with formatter
    for handler in handlers:
        handler.setFormatter(formatter)
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        handlers=handlers,
        force=True  # Override any existing configuration
    )
    return logging.getLogger(__name__)


class NHTSAEnrichmentClient:
    """Client for NHTSA Vehicle API enrichment operations."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.base_url = "https://vpic.nhtsa.dot.gov/api/vehicles/decodevinvalues"
        self.session = requests.Session()
        
        # Initialize rate limiter
        self.rate_limiter = RateLimiter(
            max_requests_per_second=config.get('api', {}).get('max_requests_per_second', 5.0)
        )
        
        # Request configuration
        self.timeout = config.get('api', {}).get('timeout', 30)
        self.max_retries = config.get('api', {}).get('max_retries', 3)
        
        # Setup session headers
        self.session.headers.update({
            'User-Agent': 'Vehicle-Data-Enrichment-Pipeline/1.0',
            'Accept': 'application/json',
        })
    
    @retry_with_backoff(max_attempts=3, base_delay=1.0, max_delay=30.0)
    def get_vehicle_info(self, vin: str) -> EnrichmentResult:
        """
        Query the NHTSA Vehicle API to get vehicle information for the given VIN.
        
        Args:
            vin (str): The Vehicle Identification Number
        
        Returns:
            EnrichmentResult: Enrichment result with cylinder count and other data
        """
        start_time = time.time()
        
        # Validate VIN format
        if not validate_vin(vin):
            return EnrichmentResult(
                vin=vin,
                cylinders=None,
                api_response_time=0.0,
                error_message="Invalid VIN format"
            )
        
        # Apply rate limiting
        self.rate_limiter.wait_if_needed()
        
        try:
            # Construct URL with proper encoding
            encoded_vin = quote(vin.strip())
            url = f"{self.base_url}/{encoded_vin}?format=json"
            
            self.logger.debug(f"Requesting data for VIN: {vin}")
            
            # Make API request
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            
            api_response_time = time.time() - start_time
            
            # Parse response
            data = response.json()
            results = data.get("Results", [])
            
            if not results:
                return EnrichmentResult(
                    vin=vin,
                    cylinders=None,
                    api_response_time=api_response_time,
                    error_message="No results returned from API"
                )
            
            vehicle_data = results[0]  # Take first result
            
            # Extract cylinder count
            cylinders_raw = vehicle_data.get("EngineCylinders", "").strip()
            cylinders = None
            if cylinders_raw:
                try:
                    cylinders = int(cylinders_raw)
                except (ValueError, TypeError):
                    self.logger.warning(f"Could not parse cylinders '{cylinders_raw}' for VIN {vin}")
            
            # Extract additional useful information
            make = vehicle_data.get("Make", "").strip() or None
            model = vehicle_data.get("Model", "").strip() or None
            year = vehicle_data.get("ModelYear", "").strip() or None
            engine_displacement = vehicle_data.get("DisplacementL", "").strip() or None
            fuel_type = vehicle_data.get("FuelTypePrimary", "").strip() or None
            
            return EnrichmentResult(
                vin=vin,
                cylinders=cylinders,
                api_response_time=api_response_time,
                make=make,
                model=model,
                year=year,
                engine_displacement=engine_displacement,
                fuel_type=fuel_type
            )
            
        except requests.exceptions.Timeout:
            api_response_time = time.time() - start_time
            return EnrichmentResult(
                vin=vin,
                cylinders=None,
                api_response_time=api_response_time,
                error_message="API request timeout"
            )
        except requests.exceptions.RequestException as e:
            api_response_time = time.time() - start_time
            return EnrichmentResult(
                vin=vin,
                cylinders=None,
                api_response_time=api_response_time,
                error_message=f"API request failed: {str(e)}"
            )
        except Exception as e:
            api_response_time = time.time() - start_time
            return EnrichmentResult(
                vin=vin,
                cylinders=None,
                api_response_time=api_response_time,
                error_message=f"Unexpected error: {str(e)}"
            )


class VehicleDataEnricher:
    """Main class for vehicle data enrichment operations."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.client = NHTSAEnrichmentClient(config)
        
        # Processing configuration
        self.batch_size = config.get('processing', {}).get('batch_size', 100)
        self.use_threading = config.get('processing', {}).get('use_threading', True)
        self.max_workers = config.get('processing', {}).get('max_workers', 5)
        self.skip_existing_cylinders = config.get('processing', {}).get('skip_existing_cylinders', True)
        
        # Output configuration
        self.output_format = config.get('output', {}).get('format', 'csv')
        self.include_api_metadata = config.get('output', {}).get('include_api_metadata', True)
    
    def load_input_data(self, input_path: str) -> pd.DataFrame:
        """Load input vehicle data from file."""
        self.logger.info(f"Loading input data from {input_path}")
        
        try:
            if input_path.endswith('.parquet'):
                if not PYARROW_AVAILABLE:
                    raise ImportError("pyarrow is required for Parquet support")
                df = pd.read_parquet(input_path)
            elif input_path.endswith('.csv'):
                df = pd.read_csv(input_path)
            else:
                # Try CSV as default
                df = pd.read_csv(input_path)
            
            self.logger.info(f"Loaded {len(df):,} rows from {input_path}")
            
            # Validate required columns
            if 'VIN' not in df.columns:
                raise ValueError("Input data must contain a 'VIN' column")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading input data: {e}")
            raise
    
    def filter_vins_for_enrichment(self, df: pd.DataFrame) -> List[str]:
        """Filter VINs that need enrichment based on configuration."""
        total_vins = len(df)
        
        # Remove rows with null/empty VINs
        df_valid_vins = df[df['VIN'].notna() & (df['VIN'].str.strip() != '')]
        
        if self.skip_existing_cylinders and 'cylinders' in df.columns:
            # Skip rows that already have cylinder data
            # Convert to string first to handle mixed types, then check for empty/null
            cylinders_str = df_valid_vins['cylinders'].astype(str)
            df_needs_enrichment = df_valid_vins[
                df_valid_vins['cylinders'].isna() | 
                (cylinders_str.str.strip() == '') |
                (cylinders_str == 'nan') |
                (cylinders_str == 'None')
            ]
            skipped_count = len(df_valid_vins) - len(df_needs_enrichment)
            self.logger.info(f"Skipping {skipped_count:,} rows that already have cylinder data")
        else:
            df_needs_enrichment = df_valid_vins
        
        # Get unique VINs to avoid duplicate API calls
        vins_to_enrich = df_needs_enrichment['VIN'].unique().tolist()
        
        self.logger.info(f"Total VINs to enrich: {len(vins_to_enrich):,} (from {total_vins:,} total rows)")
        
        return vins_to_enrich
    
    def enrich_batch_sequential(self, vins: List[str]) -> List[EnrichmentResult]:
        """Process a batch of VINs sequentially."""
        results = []
        
        for vin in vins:
            result = self.client.get_vehicle_info(vin)
            results.append(result)
            
            if result.error_message:
                self.logger.warning(f"Failed to enrich VIN {vin}: {result.error_message}", 
                                   extra={'vin': vin, 'error': result.error_message, 'response_time': result.api_response_time})
            elif result.cylinders is not None:
                self.logger.debug(f"Successfully enriched VIN {vin}: {result.cylinders} cylinders",
                                extra={'vin': vin, 'cylinders': result.cylinders, 'response_time': result.api_response_time})
        
        return results
    
    def enrich_batch_threaded(self, vins: List[str]) -> List[EnrichmentResult]:
        """Process a batch of VINs using threading."""
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all requests
            future_to_vin = {
                executor.submit(self.client.get_vehicle_info, vin): vin 
                for vin in vins
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_vin):
                result = future.result()
                results.append(result)
                
                if result.error_message:
                    self.logger.warning(f"Failed to enrich VIN {result.vin}: {result.error_message}",
                                      extra={'vin': result.vin, 'error': result.error_message, 'response_time': result.api_response_time})
                elif result.cylinders is not None:
                    self.logger.debug(f"Successfully enriched VIN {result.vin}: {result.cylinders} cylinders",
                                    extra={'vin': result.vin, 'cylinders': result.cylinders, 'response_time': result.api_response_time})
        
        return results
    
    def enrich_vins(self, vins: List[str]) -> List[EnrichmentResult]:
        """Enrich a list of VINs with API data."""
        all_results = []
        total_batches = (len(vins) + self.batch_size - 1) // self.batch_size
        
        self.logger.info(f"Processing {len(vins):,} VINs in {total_batches} batches of {self.batch_size}")
        
        # Setup progress bar if available
        if TQDM_AVAILABLE:
            progress_bar = tqdm(
                total=len(vins),
                desc="Enriching VINs",
                unit="VINs",
                unit_scale=False
            )
        else:
            progress_bar = None
        
        start_time = time.time()
        
        for batch_num in range(total_batches):
            batch_start = batch_num * self.batch_size
            batch_end = min((batch_num + 1) * self.batch_size, len(vins))
            batch_vins = vins[batch_start:batch_end]
            
            self.logger.info(f"Processing batch {batch_num + 1}/{total_batches} ({len(batch_vins)} VINs)")
            
            # Process batch
            if self.use_threading:
                batch_results = self.enrich_batch_threaded(batch_vins)
            else:
                batch_results = self.enrich_batch_sequential(batch_vins)
            
            all_results.extend(batch_results)
            
            # Update progress bar
            if progress_bar:
                progress_bar.update(len(batch_vins))
            
            # Log batch statistics
            successful = sum(1 for r in batch_results if r.cylinders is not None)
            failed = len(batch_results) - successful
            avg_response_time = sum(r.api_response_time for r in batch_results) / len(batch_results)
            
            self.logger.info(f"Batch {batch_num + 1} complete: {successful} successful, {failed} failed, "
                           f"avg response time: {avg_response_time:.2f}s", 
                           extra={'batch_id': batch_num + 1, 'success_count': successful, 'error_count': failed, 
                                'avg_response_time': avg_response_time, 'batch_size': len(batch_vins)})
        
        if progress_bar:
            progress_bar.close()
        
        total_time = time.time() - start_time
        self.logger.info(f"Enrichment complete in {total_time:.2f} seconds")
        
        return all_results
    
    def merge_enrichment_results(self, original_df: pd.DataFrame, 
                                enrichment_results: List[EnrichmentResult]) -> pd.DataFrame:
        """Merge enrichment results back into the original dataframe."""
        self.logger.info("Merging enrichment results with original data")
        
        # Convert enrichment results to dataframe
        enrichment_data = []
        for result in enrichment_results:
            row_data = {
                'VIN': result.vin,
                'api_cylinders': result.cylinders,
            }
            
            if self.include_api_metadata:
                row_data.update({
                    'api_make': result.make,
                    'api_model': result.model,
                    'api_year': result.year,
                    'api_engine_displacement': result.engine_displacement,
                    'api_fuel_type': result.fuel_type,
                    'api_response_time': result.api_response_time,
                    'api_error_message': result.error_message,
                })
            
            enrichment_data.append(row_data)
        
        enrichment_df = pd.DataFrame(enrichment_data)
        
        # Merge with original data
        enriched_df = original_df.merge(enrichment_df, on='VIN', how='left')
        
        # Update cylinders column with API data where available and original is missing
        if 'cylinders' in enriched_df.columns:
            # Create a mask for rows where original cylinders is missing/empty
            missing_cylinders = (
                enriched_df['cylinders'].isna() | 
                (enriched_df['cylinders'].astype(str).str.strip() == '') |
                (enriched_df['cylinders'] == '')
            )
            
            # Update with API data where available
            api_has_data = enriched_df['api_cylinders'].notna()
            update_mask = missing_cylinders & api_has_data
            
            enriched_df.loc[update_mask, 'cylinders'] = enriched_df.loc[update_mask, 'api_cylinders']
            
            self.logger.info(f"Updated {update_mask.sum():,} rows with API cylinder data")
        else:
            # If no existing cylinders column, create one from API data
            enriched_df['cylinders'] = enriched_df['api_cylinders']
            self.logger.info("Created new cylinders column from API data")
        
        return enriched_df
    
    def save_enriched_data(self, df: pd.DataFrame, output_path: str):
        """Save enriched data to file."""
        self.logger.info(f"Saving enriched data to {output_path}")
        
        try:
            if output_path.endswith('.parquet'):
                if not PYARROW_AVAILABLE:
                    raise ImportError("pyarrow is required for Parquet output")
                df.to_parquet(output_path, index=False)
            elif output_path.endswith('.csv'):
                df.to_csv(output_path, index=False)
            else:
                # Default to CSV
                df.to_csv(output_path, index=False)
            
            self.logger.info(f"Successfully saved {len(df):,} rows to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving enriched data: {e}")
            raise
    
    def process(self, input_path: str, output_path: str) -> Dict[str, Any]:
        """Main processing method."""
        start_time = time.time()
        
        # Load input data
        df = self.load_input_data(input_path)
        
        # Filter VINs that need enrichment
        vins_to_enrich = self.filter_vins_for_enrichment(df)
        
        if not vins_to_enrich:
            self.logger.info("No VINs need enrichment. Saving original data.")
            self.save_enriched_data(df, output_path)
            return {
                'total_rows': len(df),
                'vins_enriched': 0,
                'successful_enrichments': 0,
                'failed_enrichments': 0,
                'processing_time': time.time() - start_time
            }
        
        # Perform enrichment
        enrichment_results = self.enrich_vins(vins_to_enrich)
        
        # Merge results and save
        enriched_df = self.merge_enrichment_results(df, enrichment_results)
        self.save_enriched_data(enriched_df, output_path)
        
        # Calculate statistics
        successful_enrichments = sum(1 for r in enrichment_results if r.cylinders is not None)
        failed_enrichments = len(enrichment_results) - successful_enrichments
        
        processing_time = time.time() - start_time
        
        # Log final statistics
        self.logger.info("=" * 60)
        self.logger.info("ENRICHMENT SUMMARY")
        self.logger.info("=" * 60)
        self.logger.info(f"Total rows processed: {len(df):,}")
        self.logger.info(f"VINs enriched: {len(vins_to_enrich):,}")
        self.logger.info(f"Successful enrichments: {successful_enrichments:,}")
        self.logger.info(f"Failed enrichments: {failed_enrichments:,}")
        if len(vins_to_enrich) > 0:
            success_rate = (successful_enrichments / len(vins_to_enrich)) * 100
            self.logger.info(f"Success rate: {success_rate:.2f}%")
        self.logger.info(f"Total processing time: {processing_time:.2f} seconds")
        
        return {
            'total_rows': len(df),
            'vins_enriched': len(vins_to_enrich),
            'successful_enrichments': successful_enrichments,
            'failed_enrichments': failed_enrichments,
            'processing_time': processing_time
        }


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        raise ValueError(f"Error loading config file {config_path}: {e}")


def create_default_config() -> Dict[str, Any]:
    """Create default configuration."""
    return {
        'api': {
            'max_requests_per_second': 5.0,
            'timeout': 30,
            'max_retries': 3
        },
        'processing': {
            'batch_size': 100,
            'use_threading': True,
            'max_workers': 5,
            'skip_existing_cylinders': True
        },
        'output': {
            'format': 'csv',
            'include_api_metadata': True
        },
        'logging': {
            'level': 'INFO',
            'file': None
        }
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Enrich vehicle data with NHTSA API cylinder information",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python enrich_with_cylinders.py --input data/vehicles_cleaned.csv --output data/vehicles_enriched.csv
  python enrich_with_cylinders.py --config config/enrichment_config.yaml
  python enrich_with_cylinders.py --input data/vehicles.csv --output data/enriched.csv --batch-size 50 --max-workers 3
        """
    )
    
    parser.add_argument('--input', '-i', type=str, help='Input vehicle data file (CSV or Parquet)')
    parser.add_argument('--output', '-o', type=str, help='Output enriched data file')
    parser.add_argument('--config', '-c', type=str, help='Configuration file (YAML)')
    parser.add_argument('--batch-size', type=int, default=100, help='Batch size for processing')
    parser.add_argument('--max-workers', type=int, default=5, help='Maximum number of worker threads')
    parser.add_argument('--rate-limit', type=float, default=5.0, help='Maximum requests per second')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                       default='INFO', help='Logging level')
    parser.add_argument('--log-file', type=str, help='Log file path')
    parser.add_argument('--structured-logs', action='store_true', help='Use structured JSON logging')
    parser.add_argument('--no-threading', action='store_true', help='Disable threading')
    parser.add_argument('--include-all-cylinders', action='store_true', 
                       help='Enrich all VINs, even those with existing cylinder data')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        config = load_config(args.config)
    else:
        config = create_default_config()
    
    # Override config with command-line arguments
    if args.batch_size:
        config['processing']['batch_size'] = args.batch_size
    if args.max_workers:
        config['processing']['max_workers'] = args.max_workers
    if args.rate_limit:
        config['api']['max_requests_per_second'] = args.rate_limit
    if args.no_threading:
        config['processing']['use_threading'] = False
    if args.include_all_cylinders:
        config['processing']['skip_existing_cylinders'] = False
    if args.log_level:
        config['logging']['level'] = args.log_level
    if args.log_file:
        config['logging']['file'] = args.log_file
    if args.structured_logs:
        config['logging']['structured'] = True
    
    # Set up logging
    logger = setup_logging(
        log_level=config['logging']['level'],
        log_file=config['logging'].get('file'),
        structured=config['logging'].get('structured', False)
    )
    
    # Validate required arguments
    if not args.input and not config.get('input_path'):
        parser.error("Input file must be specified via --input or config file")
    if not args.output and not config.get('output_path'):
        parser.error("Output file must be specified via --output or config file")
    
    input_path = args.input or config.get('input_path')
    output_path = args.output or config.get('output_path')
    
    logger.info("Starting vehicle data enrichment pipeline")
    logger.info(f"Input: {input_path}")
    logger.info(f"Output: {output_path}")
    logger.info(f"Configuration: {json.dumps(config, indent=2)}")
    
    try:
        # Initialize enricher and process data
        enricher = VehicleDataEnricher(config)
        enricher.process(input_path, output_path)
        
        logger.info("Enrichment pipeline completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Enrichment pipeline failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 
