#!/usr/bin/env python3
"""
Test script for NHTSA Vehicle API enrichment

This script tests the API functionality with a few sample VINs to verify
that the enrichment pipeline is working correctly.

Usage:
    python test_api.py
    python test_api.py --vin 1HGCM82633A004352
    python test_api.py --sample-size 5
"""

import argparse
import logging
import pandas as pd

from enrich_with_cylinders import (
    NHTSAEnrichmentClient,
    create_default_config,
    setup_logging,
    validate_vin
)


def test_single_vin(vin: str) -> bool:
    """Test enrichment for a single VIN."""
    logger = logging.getLogger(__name__)
    
    # Validate VIN format first
    if not validate_vin(vin):
        logger.error(f"❌ Invalid VIN format: {vin}")
        return False
    
    config = create_default_config()
    config['api']['max_requests_per_second'] = 1.0  # Conservative for testing
    
    client = NHTSAEnrichmentClient(config)
    
    logger.info(f"Testing VIN: {vin}")
    
    try:
        result = client.get_vehicle_info(vin)
        
        if result.error_message:
            logger.error(f"❌ API Error: {result.error_message}")
            return False
        
        logger.info(f"✅ Success! Response time: {result.api_response_time:.2f}s")
        logger.info(f"   Cylinders: {result.cylinders}")
        logger.info(f"   Make: {result.make}")
        logger.info(f"   Model: {result.model}")
        logger.info(f"   Year: {result.year}")
        logger.info(f"   Engine Displacement: {result.engine_displacement}")
        logger.info(f"   Fuel Type: {result.fuel_type}")
        
        return result.cylinders is not None
        
    except Exception as e:
        logger.error(f"❌ Exception occurred: {e}")
        return False


def test_sample_vins(sample_size: int = 10) -> None:
    """Test enrichment with sample VINs from the vehicle dataset."""
    logger = logging.getLogger(__name__)
    
    # Sample VINs from different manufacturers and years
    test_vins = [
        "3GTP1VEC4EG551563",  # GMC Sierra from our data sample
        "1GCSCSE06AZ123805",  # Chevrolet Silverado from our data sample
        "1HGCM82633A004352",  # Honda Civic (example from user)
        "JTDKN3DU7A1234567",  # Toyota Prius (example)
        "1FTFW1ET5DFC12345",  # Ford F-150 (example)
        "19XFC2F59GE123456",  # Honda Civic (example)
        "1G1ZT51806F123456",  # Chevrolet Malibu (example)
        "JM1BL1V75G1234567",  # Mazda CX-5 (example)
        "2T1BURHE6FC123456",  # Toyota Corolla (example)
        "1FMCU0F70GUA12345",  # Ford Escape (example)
    ]
    
    # Use only the requested sample size
    test_vins = test_vins[:sample_size]
    
    logger.info(f"Testing {len(test_vins)} sample VINs...")
    logger.info("=" * 60)
    
    successful = 0
    failed = 0
    
    for i, vin in enumerate(test_vins, 1):
        logger.info(f"\nTest {i}/{len(test_vins)}:")
        
        if test_single_vin(vin):
            successful += 1
        else:
            failed += 1
    
    logger.info("=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total tests: {len(test_vins)}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Success rate: {(successful/len(test_vins))*100:.1f}%")


def test_with_real_data(data_path: str, sample_size: int = 5) -> None:
    """Test with real VINs from the vehicle dataset."""
    logger = logging.getLogger(__name__)
    
    try:
        logger.info(f"Loading sample data from {data_path}")
        
        # Read just enough rows to get sample VINs
        df = pd.read_csv(data_path, nrows=1000)
        
        if 'VIN' not in df.columns:
            logger.error("No VIN column found in the data")
            return
        
        # Filter valid VINs
        valid_vins = df[df['VIN'].notna() & (df['VIN'].str.strip() != '')]
        
        if len(valid_vins) == 0:
            logger.error("No valid VINs found in the data")
            return
        
        # Sample random VINs
        sample_vins = valid_vins['VIN'].sample(min(sample_size, len(valid_vins))).tolist()
        
        logger.info(f"Testing {len(sample_vins)} VINs from real data...")
        logger.info("=" * 60)
        
        successful = 0
        failed = 0
        
        for i, vin in enumerate(sample_vins, 1):
            logger.info(f"\nTest {i}/{len(sample_vins)}:")
            
            if test_single_vin(vin):
                successful += 1
            else:
                failed += 1
        
        logger.info("=" * 60)
        logger.info("REAL DATA TEST SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total tests: {len(sample_vins)}")
        logger.info(f"Successful: {successful}")
        logger.info(f"Failed: {failed}")
        logger.info(f"Success rate: {(successful/len(sample_vins))*100:.1f}%")
        
    except Exception as e:
        logger.error(f"Error testing with real data: {e}")


def main():
    """Main entry point for API testing."""
    parser = argparse.ArgumentParser(
        description="Test NHTSA Vehicle API enrichment functionality",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--vin', type=str,
                       help='Test a specific VIN')
    parser.add_argument('--sample-size', type=int, default=5,
                       help='Number of sample VINs to test')
    parser.add_argument('--data-file', type=str,
                       help='Path to vehicle data file for real data testing')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help='Logging level')
    
    args = parser.parse_args()
    
    # Set up logging
    logger = setup_logging(log_level=args.log_level)
    
    logger.info("NHTSA Vehicle API Test Suite")
    logger.info("=" * 60)
    
    try:
        if args.vin:
            # Test specific VIN
            logger.info("Testing specific VIN...")
            test_single_vin(args.vin)
            
        elif args.data_file:
            # Test with real data
            test_with_real_data(args.data_file, args.sample_size)
            
        else:
            # Test with sample VINs
            test_sample_vins(args.sample_size)
        
        logger.info("\nAPI testing completed!")
        return 0
        
    except Exception as e:
        logger.error(f"API testing failed: {e}")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main()) 
