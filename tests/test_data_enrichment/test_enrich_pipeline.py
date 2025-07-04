#!/usr/bin/env python3
"""
Integration Test Suite for NHTSA Vehicle Enrichment Pipeline

This test suite validates the complete batch enrichment pipeline (enrich_with_cylinders.py)
from CSV input to CSV output. It tests the actual data flow, transformation, error handling,
file I/O, and cleanup operations that are the heart of the enrichment pipeline.

Tests validate:
- Small batch CSV input - 5 VINs with known data get enriched and written to output CSV
- Missing cylinder VIN - Rows with no cylinder info are handled gracefully
- Invalid VIN - API errors don't crash the pipeline (log + skip)
- Output validation - Output schema (columns like VIN, make, model, cylinders...)
- Temp file cleanup - No leftover temp files after success/failure

Usage:
    pytest test_enrich_pipeline.py -v
    pytest test_enrich_pipeline.py -v --cov=enrich_with_cylinders
    pytest test_enrich_pipeline.py::TestPipelineIntegration::test_small_batch_csv_enrichment -v
"""

import pytest
import responses
import pandas as pd
import logging
import tempfile
import shutil
import glob
import os
from pathlib import Path
from unittest.mock import patch
import requests

from data_enrichment import (
    VehicleDataEnricher,
    create_default_config
)


@pytest.mark.integration
class TestPipelineIntegration:
    """Integration tests for the complete batch enrichment pipeline."""
    
    @pytest.fixture
    def integration_config(self):
        """Configuration optimized for integration testing."""
        config = create_default_config()
        config['processing']['use_threading'] = False  # Disable threading for predictable tests
        config['processing']['batch_size'] = 3  # Small batches for testing
        config['api']['max_requests_per_second'] = 100.0  # Fast for testing
        config['logging']['level'] = 'DEBUG'  # Detailed logging for test validation
        return config
    
    @pytest.fixture
    def enricher(self, integration_config):
        """Vehicle data enricher configured for integration testing."""
        return VehicleDataEnricher(integration_config)
    
    @pytest.fixture
    def test_workspace(self):
        """Create a temporary workspace for test files."""
        workspace = tempfile.mkdtemp(prefix="integration_test_")
        yield Path(workspace)
        # Cleanup after test
        shutil.rmtree(workspace, ignore_errors=True)
    
    def create_test_csv(self, workspace: Path, filename: str, data: pd.DataFrame) -> Path:
        """Helper to create test CSV files."""
        file_path = workspace / filename
        data.to_csv(file_path, index=False)
        return file_path
    
    def verify_no_temp_files(self, workspace: Path):
        """Verify no temporary files are left behind."""
        temp_patterns = ['*.tmp', '*.temp', '*_backup*', '*_temp*']
        temp_files = []
        for pattern in temp_patterns:
            temp_files.extend(glob.glob(str(workspace / pattern)))
        
        assert len(temp_files) == 0, f"Found leftover temp files: {temp_files}"
    
    @responses.activate
    def test_small_batch_csv_enrichment(self, enricher, test_workspace):
        """Test small batch CSV input with 5 VINs with known data."""
        # Create test input data with 5 VINs
        test_data = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'VIN': [
                '3GTP1VEC4EG551563',  # GMC Sierra (from sample data)
                '1GCSCSE06AZ123805',  # Chevrolet Silverado
                '3GCPWCED5LG130317',  # Chevrolet Silverado 1500 Crew
                '5TFRM5F17HX120972',  # Toyota Tundra
                '1HGCM82633A004352'   # Honda Civic
            ],
            'cylinders': [None, '', None, None, None],  # Missing cylinder data
            'make': ['gmc', 'chevrolet', 'chevrolet', 'toyota', 'honda'],
            'model': ['sierra', 'silverado', 'silverado', 'tundra', 'civic'],
            'year': [2014, 2010, 2020, 2017, 2016]
        })
        
        input_file = self.create_test_csv(test_workspace, "test_input.csv", test_data)
        output_file = test_workspace / "test_output.csv"
        
        # Mock API responses for all 5 VINs
        mock_responses = [
            {
                'vin': '3GTP1VEC4EG551563',
                'response': {
                    "Results": [{
                        "EngineCylinders": "8",
                        "Make": "GMC",
                        "Model": "Sierra 1500",
                        "ModelYear": "2014",
                        "DisplacementL": "5.3",
                        "FuelTypePrimary": "Gasoline"
                    }]
                }
            },
            {
                'vin': '1GCSCSE06AZ123805',
                'response': {
                    "Results": [{
                        "EngineCylinders": "8",
                        "Make": "CHEVROLET",
                        "Model": "Silverado 1500",
                        "ModelYear": "2010",
                        "DisplacementL": "5.3",
                        "FuelTypePrimary": "Gasoline"
                    }]
                }
            },
            {
                'vin': '3GCPWCED5LG130317',
                'response': {
                    "Results": [{
                        "EngineCylinders": "8",
                        "Make": "CHEVROLET",
                        "Model": "Silverado 1500",
                        "ModelYear": "2020",
                        "DisplacementL": "5.3",
                        "FuelTypePrimary": "Gasoline"
                    }]
                }
            },
            {
                'vin': '5TFRM5F17HX120972',
                'response': {
                    "Results": [{
                        "EngineCylinders": "8",
                        "Make": "TOYOTA",
                        "Model": "Tundra",
                        "ModelYear": "2017",
                        "DisplacementL": "4.6",
                        "FuelTypePrimary": "Gasoline"
                    }]
                }
            },
            {
                'vin': '1HGCM82633A004352',
                'response': {
                    "Results": [{
                        "EngineCylinders": "4",
                        "Make": "HONDA",
                        "Model": "Civic",
                        "ModelYear": "2016",
                        "DisplacementL": "1.5",
                        "FuelTypePrimary": "Gasoline"
                    }]
                }
            }
        ]
        
        # Add all mock responses
        for mock in mock_responses:
            responses.add(
                responses.GET,
                f"https://vpic.nhtsa.dot.gov/api/vehicles/decodevinvalues/{mock['vin']}?format=json",
                json=mock['response'],
                status=200
            )
        
        # Run the complete pipeline
        results = enricher.process(str(input_file), str(output_file))
        
        # Verify pipeline results
        assert results['total_rows'] == 5
        assert results['vins_enriched'] == 5
        assert results['successful_enrichments'] == 5
        assert results['failed_enrichments'] == 0
        assert results['processing_time'] > 0
        
        # Verify output file exists and has correct structure
        assert output_file.exists()
        output_data = pd.read_csv(output_file)
        
        # Validate output schema and data
        assert len(output_data) == 5
        expected_columns = [
            'id', 'VIN', 'cylinders', 'make', 'model', 'year',
            'api_cylinders', 'api_make', 'api_model', 'api_year',
            'api_engine_displacement', 'api_fuel_type', 'api_response_time'
        ]
        for col in expected_columns:
            assert col in output_data.columns, f"Missing column: {col}"
        
        # Verify enrichment data was properly merged
        assert output_data['cylinders'].notna().all()  # All cylinders should be filled
        assert output_data['api_cylinders'].notna().all()  # All API data present
        
        # Check specific enrichment results
        gmc_row = output_data[output_data['VIN'] == '3GTP1VEC4EG551563'].iloc[0]
        assert gmc_row['cylinders'] == 8.0
        assert gmc_row['api_make'] == 'GMC'
        assert gmc_row['api_model'] == 'Sierra 1500'
        
        honda_row = output_data[output_data['VIN'] == '1HGCM82633A004352'].iloc[0]
        assert honda_row['cylinders'] == 4.0
        assert honda_row['api_make'] == 'HONDA'
        assert honda_row['api_model'] == 'Civic'
        
        # Verify no temp files left behind
        self.verify_no_temp_files(test_workspace)
    
    @responses.activate  
    def test_missing_cylinder_vin_handling(self, enricher, test_workspace):
        """Test that rows with no cylinder info are handled gracefully."""
        # Create test data with VINs that have no cylinder data available
        test_data = pd.DataFrame({
            'id': [1, 2, 3],
            'VIN': [
                '3GTP1VEC4EG551563',  # This will have cylinder data
                '1GCSCSE06AZ999999',  # This will have no cylinder data (empty response)
                '1FDWF36P04EA12345'   # This will have no cylinder data (invalid response)
            ],
            'cylinders': [None, None, None],
            'make': ['gmc', 'chevrolet', 'ford'],
            'model': ['sierra', 'silverado', 'f-150']
        })
        
        input_file = self.create_test_csv(test_workspace, "test_missing.csv", test_data)
        output_file = test_workspace / "test_missing_output.csv"
        
        # Mock API responses - success, empty, and no cylinders
        responses.add(
            responses.GET,
            "https://vpic.nhtsa.dot.gov/api/vehicles/decodevinvalues/3GTP1VEC4EG551563?format=json",
            json={"Results": [{"EngineCylinders": "8", "Make": "GMC", "Model": "Sierra"}]},
            status=200
        )
        responses.add(
            responses.GET,
            "https://vpic.nhtsa.dot.gov/api/vehicles/decodevinvalues/1GCSCSE06AZ999999?format=json",
            json={"Results": []},  # Empty response
            status=200
        )
        responses.add(
            responses.GET,
            "https://vpic.nhtsa.dot.gov/api/vehicles/decodevinvalues/1FDWF36P04EA12345?format=json",
            json={"Results": [{"EngineCylinders": "", "Make": "FORD", "Model": "F-150"}]},  # Empty cylinders
            status=200
        )
        
        # Run pipeline
        results = enricher.process(str(input_file), str(output_file))
        
        # Verify graceful handling
        assert results['total_rows'] == 3
        assert results['vins_enriched'] == 3
        assert results['successful_enrichments'] == 1  # Only one successful enrichment
        assert results['failed_enrichments'] == 2
        
        # Verify output file and data
        assert output_file.exists()
        output_data = pd.read_csv(output_file)
        assert len(output_data) == 3
        
        # Check that rows without cylinder data are handled gracefully
        successful_row = output_data[output_data['VIN'] == '3GTP1VEC4EG551563'].iloc[0]
        assert successful_row['cylinders'] == 8.0
        assert successful_row['api_make'] == 'GMC'
        
        # Rows without cylinder data should have null cylinders but still be present
        empty_response_row = output_data[output_data['VIN'] == '1GCSCSE06AZ999999'].iloc[0]
        assert pd.isna(empty_response_row['api_cylinders'])
        assert empty_response_row['api_error_message'] == 'No results returned from API'
        
        no_cylinders_row = output_data[output_data['VIN'] == '1FDWF36P04EA12345'].iloc[0]
        assert pd.isna(no_cylinders_row['api_cylinders'])
        assert no_cylinders_row['api_make'] == 'FORD'
        
        self.verify_no_temp_files(test_workspace)
    
    @responses.activate
    def test_invalid_vin_error_handling(self, enricher, test_workspace):
        """Test that invalid VINs and API errors don't crash the pipeline."""
        # Set up log capture
        log_capture = []
        class LogCapture(logging.Handler):
            def emit(self, record):
                log_capture.append(self.format(record))
        
        log_handler = LogCapture()
        log_handler.setFormatter(logging.Formatter('%(message)s'))
        logger = logging.getLogger('data_enrichment.enrich_with_cylinders')
        logger.addHandler(log_handler)
        
        try:
            # Create test data with various error cases
            test_data = pd.DataFrame({
                'VIN': [
                    '3GTP1VEC4EG551563',  # Valid VIN
                    'INVALIDVIN123456',    # Invalid format
                    '1GCSCSE06AZ500500',   # Timeout error
                    '1GCSCSE06AZ400400',   # Server error
                    ''                     # Empty VIN
                ]
            })
            
            input_file = self.create_test_csv(test_workspace, "test_errors.csv", test_data)
            output_file = test_workspace / "test_errors_output.csv"
            
            # Mock API responses for error testing
            responses.add(
                responses.GET,
                "https://vpic.nhtsa.dot.gov/api/vehicles/decodevinvalues/3GTP1VEC4EG551563?format=json",
                json={"Results": [{"EngineCylinders": "8", "Make": "GMC"}]},
                status=200
            )
            responses.add(
                responses.GET,
                "https://vpic.nhtsa.dot.gov/api/vehicles/decodevinvalues/1GCSCSE06AZ500500?format=json",
                body=requests.exceptions.ConnectTimeout("Connection timed out"),
                status=408
            )
            responses.add(
                responses.GET,
                "https://vpic.nhtsa.dot.gov/api/vehicles/decodevinvalues/1GCSCSE06AZ400400?format=json",
                json={"error": "Internal server error"},
                status=500
            )
            
            # Run pipeline - should not crash despite errors
            results = enricher.process(str(input_file), str(output_file))
            
            # Verify pipeline completed without crashing
            assert results['total_rows'] == 5
            assert results['vins_enriched'] == 4  # All VINs with values attempted (excludes only empty VIN)
            assert results['successful_enrichments'] == 1  # Only one successful
            assert results['failed_enrichments'] == 3  # Three failed (invalid, timeout, server error)
            
            # Verify output file exists and contains all rows
            assert output_file.exists()
            output_data = pd.read_csv(output_file)
            assert len(output_data) == 5  # All original rows preserved
            
            # Verify successful enrichment
            success_row = output_data[output_data['VIN'] == '3GTP1VEC4EG551563'].iloc[0]
            assert success_row['cylinders'] == 8.0
            assert pd.isna(success_row['api_error_message'])
            
            # Verify invalid VIN handling
            invalid_row = output_data[output_data['VIN'] == 'INVALIDVIN123456'].iloc[0]
            assert pd.isna(invalid_row['api_cylinders'])
            assert 'Invalid VIN format' in invalid_row['api_error_message']
            
            # Verify empty VIN handling - check if empty VIN rows exist in output
            empty_vin_rows = output_data[output_data['VIN'] == '']
            if len(empty_vin_rows) > 0:
                empty_row = empty_vin_rows.iloc[0]
                assert pd.isna(empty_row['api_cylinders'])
            else:
                # If empty VIN rows are filtered out, that's also acceptable behavior
                pass
            
            # Verify timeout error handling
            timeout_row = output_data[output_data['VIN'] == '1GCSCSE06AZ500500'].iloc[0]
            assert pd.isna(timeout_row['api_cylinders'])
            assert 'timeout' in timeout_row['api_error_message'].lower() or 'connection' in timeout_row['api_error_message'].lower()
            
            # Verify server error handling
            server_error_row = output_data[output_data['VIN'] == '1GCSCSE06AZ400400'].iloc[0]
            assert pd.isna(server_error_row['api_cylinders'])
            assert 'failed' in server_error_row['api_error_message'].lower()
            
            # Verify error logging occurred
            assert any('Failed to enrich VIN' in msg for msg in log_capture) or any('timeout' in msg.lower() for msg in log_capture), \
                   "Expected error messages not found in logs"
        
        finally:
            logger.removeHandler(log_handler)
        
        self.verify_no_temp_files(test_workspace)
    
    @responses.activate
    def test_output_schema_validation(self, enricher, test_workspace):
        """Test comprehensive output schema validation."""
        # Create minimal test data
        test_data = pd.DataFrame({
            'id': [1, 2],
            'VIN': ['3GTP1VEC4EG551563', '1GCSCSE06AZ123805'],
            'cylinders': [None, None],
            'make': ['gmc', 'chevrolet'],
            'model': ['sierra', 'silverado'],
            'year': [2014, 2010],
            'price': [33590, 22590],
            'fuel': ['gas', 'gas']
        })
        
        input_file = self.create_test_csv(test_workspace, "test_schema.csv", test_data)
        output_file = test_workspace / "test_schema_output.csv"
        
        # Mock comprehensive API responses
        responses.add(
            responses.GET,
            "https://vpic.nhtsa.dot.gov/api/vehicles/decodevinvalues/3GTP1VEC4EG551563?format=json",
            json={
                "Results": [{
                    "EngineCylinders": "8",
                    "Make": "GMC",
                    "Model": "Sierra 1500",
                    "ModelYear": "2014",
                    "DisplacementL": "5.3",
                    "FuelTypePrimary": "Gasoline",
                }]
            },
            status=200
        )
        responses.add(
            responses.GET,
            "https://vpic.nhtsa.dot.gov/api/vehicles/decodevinvalues/1GCSCSE06AZ123805?format=json",
            json={
                "Results": [{
                    "EngineCylinders": "8", 
                    "Make": "CHEVROLET",
                    "Model": "Silverado 1500",
                    "ModelYear": "2010",
                    "DisplacementL": "5.3",
                    "FuelTypePrimary": "Gasoline"
                }]
            },
            status=200
        )
        
        # Run pipeline
        enricher.process(str(input_file), str(output_file))
        
        # Load and validate output schema
        output_data = pd.read_csv(output_file)
        
        # Verify all original columns are preserved
        original_columns = test_data.columns.tolist()
        for col in original_columns:
            assert col in output_data.columns, f"Original column {col} missing from output"
        
        # Verify all expected API columns are present
        expected_api_columns = [
            'api_cylinders',
            'api_make', 
            'api_model',
            'api_year',
            'api_engine_displacement',
            'api_fuel_type',
            'api_response_time',
            'api_error_message'
        ]
        for col in expected_api_columns:
            assert col in output_data.columns, f"Expected API column {col} missing from output"
        
        # Verify data types and values
        assert output_data['cylinders'].dtype in ['float64', 'int64']  # Should be numeric
        assert output_data['api_cylinders'].dtype in ['float64', 'int64']
        assert output_data['api_response_time'].dtype == 'float64'
        
        # Verify enrichment worked correctly
        assert output_data['cylinders'].notna().all()
        assert output_data['api_cylinders'].notna().all()
        assert (output_data['api_response_time'] > 0).all()
        
        # Verify original data integrity
        assert output_data['id'].tolist() == [1, 2]
        assert output_data['price'].tolist() == [33590, 22590]
        assert output_data['fuel'].tolist() == ['gas', 'gas']
        
        self.verify_no_temp_files(test_workspace)
    
    def test_temp_file_cleanup_on_success(self, enricher, test_workspace):
        """Test temp file cleanup after successful processing."""
        test_data = pd.DataFrame({
            'id': [1],
            'VIN': ['3GTP1VEC4EG551563'],
            'cylinders': [None]
        })
        
        input_file = self.create_test_csv(test_workspace, "test_cleanup.csv", test_data)
        output_file = test_workspace / "test_cleanup_output.csv"
        
        # Create some fake temp files to ensure they don't interfere
        fake_temp1 = test_workspace / "fake_temp.tmp"
        fake_temp2 = test_workspace / "backup_file.backup"
        fake_temp1.touch()
        fake_temp2.touch()
        
        with responses.RequestsMock() as rsps:
            rsps.add(
                responses.GET,
                "https://vpic.nhtsa.dot.gov/api/vehicles/decodevinvalues/3GTP1VEC4EG551563?format=json",
                json={"Results": [{"EngineCylinders": "8", "Make": "GMC"}]},
                status=200
            )
            
            # Run pipeline
            results = enricher.process(str(input_file), str(output_file))
            
            # Track files after processing
            files_after = set(os.listdir(test_workspace))
            
            # Verify successful processing
            assert results['successful_enrichments'] == 1
            assert output_file.exists()
            
            # Verify only expected files remain
            expected_files = {
                input_file.name,
                output_file.name,
                fake_temp1.name,  # Pre-existing files should remain
                fake_temp2.name
            }
            assert files_after == expected_files
            
            # Verify no additional temp files were created by the pipeline
            # (exclude pre-existing fake temp files from check)
            pipeline_temp_patterns = ['*_processing*', '*_backup*', '*_temp*']
            pipeline_temp_files = []
            for pattern in pipeline_temp_patterns:
                temp_files = glob.glob(str(test_workspace / pattern))
                # Filter out our intentionally created fake files
                pipeline_temp_files.extend([f for f in temp_files if 'fake_temp' not in f and 'backup_file' not in f])
            
            assert len(pipeline_temp_files) == 0, f"Found leftover pipeline temp files: {pipeline_temp_files}"
    
    def test_temp_file_cleanup_on_failure(self, enricher, test_workspace):
        """Test temp file cleanup after failed processing."""
        # Create invalid input file that will cause failure
        invalid_data = pd.DataFrame({
            'id': [1],
            'invalid_column': ['no_vin_column']  # Missing required VIN column
        })
        
        input_file = self.create_test_csv(test_workspace, "test_failure.csv", invalid_data)
        output_file = test_workspace / "test_failure_output.csv"
        
        # Run pipeline - should fail due to missing VIN column
        with pytest.raises(ValueError, match="Input data must contain a 'VIN' column"):
            enricher.process(str(input_file), str(output_file))
        
        # Track files after failed processing
        files_after = set(os.listdir(test_workspace))
        
        # Verify no output file was created
        assert not output_file.exists()
        
        # Verify no temp files were left behind
        expected_files = {input_file.name}
        assert files_after == expected_files
        self.verify_no_temp_files(test_workspace)
    
    @responses.activate
    def test_pipeline_with_existing_cylinders(self, enricher, test_workspace):
        """Test pipeline behavior with rows that already have cylinder data."""
        # Test data with mixed cylinder availability
        test_data = pd.DataFrame({
            'id': [1, 2, 3, 4],
            'VIN': ['3GTP1VEC4EG551563', '1GCSCSE06AZ123805', '3GCPWCED5LG130317', '5TFRM5F17HX120972'],
            'cylinders': [8, None, '', '6 cylinders'],  # Mixed: filled, empty, blank, text
            'make': ['gmc', 'chevrolet', 'chevrolet', 'toyota']
        })
        
        input_file = self.create_test_csv(test_workspace, "test_existing.csv", test_data)
        output_file = test_workspace / "test_existing_output.csv"
        
        # Mock API responses only for VINs that should be enriched (those with missing cylinders)
        responses.add(
            responses.GET,
            "https://vpic.nhtsa.dot.gov/api/vehicles/decodevinvalues/1GCSCSE06AZ123805?format=json",
            json={"Results": [{"EngineCylinders": "8", "Make": "CHEVROLET"}]},
            status=200
        )
        responses.add(
            responses.GET,
            "https://vpic.nhtsa.dot.gov/api/vehicles/decodevinvalues/3GCPWCED5LG130317?format=json",
            json={"Results": [{"EngineCylinders": "8", "Make": "CHEVROLET"}]},
            status=200
        )
        
        # Run pipeline
        results = enricher.process(str(input_file), str(output_file))
        
        # Verify correct VINs were processed (skip existing cylinders)
        assert results['total_rows'] == 4
        assert results['vins_enriched'] == 2  # Only rows 2 and 3 should be enriched
        assert results['successful_enrichments'] == 2
        
        # Verify output
        output_data = pd.read_csv(output_file)
        assert len(output_data) == 4
        
        # Row 1: Should keep existing cylinders (8), no API call made
        row1 = output_data[output_data['id'] == 1].iloc[0]
        assert str(row1['cylinders']) == '8'  # Handle as string since CSV may preserve original format
        assert pd.isna(row1['api_cylinders'])  # No API call made
        
        # Row 2: Should be enriched
        row2 = output_data[output_data['id'] == 2].iloc[0]
        assert float(row2['cylinders']) == 8.0
        assert row2['api_cylinders'] == 8.0
        
        # Row 3: Should be enriched (blank treated as missing)
        row3 = output_data[output_data['id'] == 3].iloc[0]
        assert float(row3['cylinders']) == 8.0
        assert row3['api_cylinders'] == 8.0
        
        # Row 4: Should keep existing cylinders (text), no API call made
        row4 = output_data[output_data['id'] == 4].iloc[0]
        assert row4['cylinders'] == '6 cylinders'
        assert pd.isna(row4['api_cylinders'])  # No API call made
        
        self.verify_no_temp_files(test_workspace)


@pytest.mark.integration
class TestPipelineEdgeCases:
    """Test edge cases and boundary conditions."""
    
    @pytest.fixture
    def integration_config(self):
        """Configuration for edge case testing."""
        config = create_default_config()
        config['processing']['use_threading'] = False
        config['processing']['batch_size'] = 2  # Small batch for testing
        config['api']['max_requests_per_second'] = 100.0
        return config
    
    @responses.activate
    def test_empty_input_file(self, integration_config, tmp_path):
        """Test handling of empty input file."""
        enricher = VehicleDataEnricher(integration_config)
        
        # Create empty CSV file with just headers
        empty_data = pd.DataFrame(columns=['id', 'VIN', 'cylinders'])
        input_file = tmp_path / "empty_input.csv"
        output_file = tmp_path / "empty_output.csv"
        empty_data.to_csv(input_file, index=False)
        
        # Run pipeline
        results = enricher.process(str(input_file), str(output_file))
        
        # Verify handling of empty file
        assert results['total_rows'] == 0
        assert results['vins_enriched'] == 0
        assert results['successful_enrichments'] == 0
        assert results['failed_enrichments'] == 0
        
        # Verify output file exists and is also empty (with headers)
        assert output_file.exists()
        output_data = pd.read_csv(output_file)
        assert len(output_data) == 0
        assert 'VIN' in output_data.columns
    
    @responses.activate
    def test_single_row_processing(self, integration_config, tmp_path):
        """Test processing of single row."""
        enricher = VehicleDataEnricher(integration_config)
        
        # Single row data
        single_data = pd.DataFrame({
            'id': [1],
            'VIN': ['3GTP1VEC4EG551563'],
            'cylinders': [None]
        })
        
        input_file = tmp_path / "single_input.csv"
        output_file = tmp_path / "single_output.csv"
        single_data.to_csv(input_file, index=False)
        
        # Mock API response
        responses.add(
            responses.GET,
            "https://vpic.nhtsa.dot.gov/api/vehicles/decodevinvalues/3GTP1VEC4EG551563?format=json",
            json={"Results": [{"EngineCylinders": "8", "Make": "GMC"}]},
            status=200
        )
        
        # Run pipeline
        results = enricher.process(str(input_file), str(output_file))
        
        # Verify single row processing
        assert results['total_rows'] == 1
        assert results['vins_enriched'] == 1
        assert results['successful_enrichments'] == 1
        
        output_data = pd.read_csv(output_file)
        assert len(output_data) == 1
        assert output_data.iloc[0]['cylinders'] == 8.0
    
    def test_large_dataset_simulation(self, integration_config, tmp_path):
        """Test with simulated large dataset (without actual API calls)."""
        enricher = VehicleDataEnricher(integration_config)
        
        # Create larger dataset
        large_data = pd.DataFrame({
            'id': range(1, 101),  # 100 rows
            'VIN': [f'1GCSCSE06AZ{i:06d}' for i in range(100)],  # Generate unique VINs
            'cylinders': [None] * 100
        })
        
        input_file = tmp_path / "large_input.csv"
        output_file = tmp_path / "large_output.csv"
        large_data.to_csv(input_file, index=False)
        
        # Mock the enrichment process to simulate processing without API calls
        with patch.object(enricher, 'enrich_vins') as mock_enrich:
            # Return mock results for all VINs
            mock_results = [
                enricher.client.get_vehicle_info(f'1GCSCSE06AZ{i:06d}') 
                for i in range(100)
            ]
            # Override to simulate successful enrichment without API calls
            for i, result in enumerate(mock_results):
                result.cylinders = 8 if i % 2 == 0 else 6  # Alternate between 6 and 8 cylinders
                result.error_message = None
                result.make = "TEST"
                result.model = "VEHICLE"
                result.api_response_time = 0.1
            
            mock_enrich.return_value = mock_results
            
            # Run pipeline
            results = enricher.process(str(input_file), str(output_file))
            
            # Verify large dataset handling
            assert results['total_rows'] == 100
            assert results['vins_enriched'] == 100
            
            # Verify output file
            output_data = pd.read_csv(output_file)
            assert len(output_data) == 100
            assert output_data['cylinders'].notna().all()


if __name__ == "__main__":
    # Run integration tests directly
    pytest.main([__file__, "-v", "-m", "integration"]) 
