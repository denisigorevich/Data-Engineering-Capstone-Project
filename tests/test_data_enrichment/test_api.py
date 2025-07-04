#!/usr/bin/env python3
"""
Test script for NHTSA Vehicle API enrichment

This script tests the API functionality with a few sample VINs to verify
that the enrichment pipeline is working correctly.
"""

import logging
import pandas as pd
import pytest

from data_enrichment import (
    create_default_config
)
from data_enrichment.enrich_with_cylinders import (
    NHTSAEnrichmentClient,
    validate_vin
)


class TestAPIFunctionality:
    """Test suite for API functionality."""
    
    @pytest.fixture
    def config(self):
        """Test configuration fixture."""
        config = create_default_config()
        config['api']['max_requests_per_second'] = 1.0  # Conservative for testing
        return config
    
    @pytest.fixture
    def client(self, config):
        """Test API client fixture."""
        return NHTSAEnrichmentClient(config)
    
    @pytest.fixture
    def sample_vins(self):
        """Sample VINs for testing."""
        return [
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

    def test_single_vin(self, client, sample_vins):
        """Test enrichment for a single VIN."""
        logger = logging.getLogger(__name__)
        vin = sample_vins[0]  # Use first sample VIN
        
        # Validate VIN format first
        assert validate_vin(vin), f"Invalid VIN format: {vin}"
        
        logger.info(f"Testing VIN: {vin}")
        
        try:
            result = client.get_vehicle_info(vin)
            
            assert not result.error_message, f"API Error: {result.error_message}"
            
            logger.info(f"✅ Success! Response time: {result.api_response_time:.2f}s")
            logger.info(f"   Cylinders: {result.cylinders}")
            logger.info(f"   Make: {result.make}")
            logger.info(f"   Model: {result.model}")
            logger.info(f"   Year: {result.year}")
            logger.info(f"   Engine Displacement: {result.engine_displacement}")
            logger.info(f"   Fuel Type: {result.fuel_type}")
            
            assert result.cylinders is not None, "No cylinder data returned"
            
        except Exception as e:
            logger.error(f"❌ Exception occurred: {e}")
            pytest.fail(f"Test failed with exception: {e}")

    def test_sample_vins(self, client, sample_vins):
        """Test enrichment with sample VINs."""
        logger = logging.getLogger(__name__)
        
        # Use only first 5 VINs for faster testing
        test_vins = sample_vins[:5]
        
        logger.info(f"Testing {len(test_vins)} sample VINs...")
        logger.info("=" * 60)
        
        successful = 0
        failed = 0
        
        for i, vin in enumerate(test_vins, 1):
            logger.info(f"\nTest {i}/{len(test_vins)}:")
            
            try:
                result = client.get_vehicle_info(vin)
                if result.cylinders is not None and not result.error_message:
                    successful += 1
                else:
                    failed += 1
            except Exception:
                failed += 1
        
        logger.info("=" * 60)
        logger.info("TEST SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total tests: {len(test_vins)}")
        logger.info(f"Successful: {successful}")
        logger.info(f"Failed: {failed}")
        logger.info(f"Success rate: {(successful/len(test_vins))*100:.1f}%")
        
        # Assert reasonable success rate (80% or better)
        assert successful/len(test_vins) >= 0.8, "Success rate below 80%"

    def test_with_real_data(self, client, tmp_path):
        """Test with real VINs from the vehicle dataset."""
        logger = logging.getLogger(__name__)
        
        # Create a sample data file
        data = pd.DataFrame({
            'VIN': [
                "3GTP1VEC4EG551563",
                "1GCSCSE06AZ123805",
                "1HGCM82633A004352"
            ]
        })
        data_path = tmp_path / "test_data.csv"
        data.to_csv(data_path, index=False)
        
        try:
            logger.info(f"Loading sample data from {data_path}")
            df = pd.read_csv(data_path)
            
            assert 'VIN' in df.columns, "No VIN column found in the data"
            
            # Filter valid VINs
            valid_vins = df[df['VIN'].notna() & (df['VIN'].str.strip() != '')]
            
            assert len(valid_vins) > 0, "No valid VINs found in the data"
            
            # Test each VIN
            successful = 0
            failed = 0
            
            for i, vin in enumerate(valid_vins['VIN'], 1):
                logger.info(f"\nTest {i}/{len(valid_vins)}:")
                
                try:
                    result = client.get_vehicle_info(vin)
                    if result.cylinders is not None and not result.error_message:
                        successful += 1
                    else:
                        failed += 1
                except Exception:
                    failed += 1
            
            logger.info("=" * 60)
            logger.info("REAL DATA TEST SUMMARY")
            logger.info("=" * 60)
            logger.info(f"Total tests: {len(valid_vins)}")
            logger.info(f"Successful: {successful}")
            logger.info(f"Failed: {failed}")
            logger.info(f"Success rate: {(successful/len(valid_vins))*100:.1f}%")
            
            # Assert reasonable success rate (80% or better)
            assert successful/len(valid_vins) >= 0.8, "Success rate below 80%"
            
        except Exception as e:
            logger.error(f"Error testing with real data: {e}")
            pytest.fail(f"Test failed with exception: {e}") 
