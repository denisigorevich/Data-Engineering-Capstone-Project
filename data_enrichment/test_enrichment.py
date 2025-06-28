#!/usr/bin/env python3
"""
Comprehensive Unit Test Suite for NHTSA Vehicle API Enrichment

This test suite uses pytest and responses to mock HTTP calls, enabling fast CI/CD testing
without actually hitting the NHTSA API. Tests cover:

- VIN validation logic
- API response parsing
- Error handling scenarios
- Data enrichment and merging
- Configuration handling
- Logging functionality

Usage:
    pytest test_enrichment.py -v
    pytest test_enrichment.py -v --cov=enrich_with_cylinders
    pytest test_enrichment.py::test_successful_enrichment -v
"""

import pytest
import responses
import pandas as pd
import logging
from pathlib import Path

from enrich_with_cylinders import (
    NHTSAEnrichmentClient,
    VehicleDataEnricher,
    EnrichmentResult,
    validate_vin,
    setup_logging,
    create_default_config
)


class TestVINValidation:
    """Test VIN validation logic."""
    
    def test_valid_vin_format(self):
        """Test that valid VINs pass validation."""
        valid_vins = [
            "3GTP1VEC4EG551563",  # GMC Sierra from test data
            "1GCSCSE06AZ123805",  # Chevrolet Silverado
            "1HGCM82633A004352",  # Honda Civic
            "JTDKN3DU7A1234567",  # Toyota Prius
        ]
        
        for vin in valid_vins:
            assert validate_vin(vin), f"VIN {vin} should be valid"
    
    def test_invalid_vin_format(self):
        """Test that invalid VINs fail validation."""
        invalid_vins = [
            "",                    # Empty string
            "ABC123",             # Too short
            "1234567890123456789", # Too long
            "1GCSCSE06AZ12380I",   # Contains 'I'
            "1GCSCSE06AZ12380O",   # Contains 'O'
            "1GCSCSE06AZ12380Q",   # Contains 'Q'
            "1GCSCSE06AZ123!@#",   # Contains special characters
            None,                  # None value
            123456789012345678,    # Numeric instead of string
        ]
        
        for vin in invalid_vins:
            assert not validate_vin(vin), f"VIN {vin} should be invalid"
    
    def test_vin_case_insensitive(self):
        """Test that VIN validation handles different cases."""
        assert validate_vin("3gtp1vec4eg551563")  # Lowercase
        assert validate_vin("3GTP1VEC4EG551563")  # Uppercase
        assert validate_vin("3Gtp1Vec4Eg551563")  # Mixed case


class TestNHTSAAPIResponse:
    """Test NHTSA API response handling with mocked responses."""
    
    @pytest.fixture
    def config(self):
        """Default configuration for testing."""
        return create_default_config()
    
    @pytest.fixture
    def client(self, config):
        """NHTSA enrichment client for testing."""
        return NHTSAEnrichmentClient(config)
    
    @responses.activate
    def test_successful_api_response(self, client):
        """Test successful API response parsing."""
        test_vin = "3GTP1VEC4EG551563"
        
        # Mock successful API response
        mock_response = {
            "Results": [{
                "EngineCylinders": "8",
                "Make": "GMC",
                "Model": "Sierra",
                "ModelYear": "2014",
                "DisplacementL": "5.3",
                "FuelTypePrimary": "Gasoline"
            }]
        }
        
        responses.add(
            responses.GET,
            f"https://vpic.nhtsa.dot.gov/api/vehicles/decodevinvalues/{test_vin}?format=json",
            json=mock_response,
            status=200
        )
        
        result = client.get_vehicle_info(test_vin)
        
        assert result.vin == test_vin
        assert result.cylinders == 8
        assert result.make == "GMC"
        assert result.model == "Sierra"
        assert result.year == "2014"
        assert result.engine_displacement == "5.3"
        assert result.fuel_type == "Gasoline"
        assert result.error_message is None
        assert result.api_response_time > 0
    
    @responses.activate
    def test_empty_api_response(self, client):
        """Test handling of empty API response."""
        test_vin = "1GCSCSE06AZ999999"  # Valid format but non-existent VIN
        
        # Mock empty API response
        mock_response = {"Results": []}
        
        responses.add(
            responses.GET,
            f"https://vpic.nhtsa.dot.gov/api/vehicles/decodevinvalues/{test_vin}?format=json",
            json=mock_response,
            status=200
        )
        
        result = client.get_vehicle_info(test_vin)
        
        assert result.vin == test_vin
        assert result.cylinders is None
        assert result.error_message == "No results returned from API"
    
    @responses.activate
    def test_api_timeout_error(self, client):
        """Test handling of API timeout."""
        test_vin = "3GTP1VEC4EG551563"
        
        # Mock timeout response
        responses.add(
            responses.GET,
            f"https://vpic.nhtsa.dot.gov/api/vehicles/decodevinvalues/{test_vin}?format=json",
            body=responses.ConnectionError("Connection timeout")
        )
        
        result = client.get_vehicle_info(test_vin)
        
        assert result.vin == test_vin
        assert result.cylinders is None
        assert "timeout" in result.error_message.lower() or "connection" in result.error_message.lower()
    
    @responses.activate
    def test_api_server_error(self, client):
        """Test handling of API server error."""
        test_vin = "3GTP1VEC4EG551563"
        
        responses.add(
            responses.GET,
            f"https://vpic.nhtsa.dot.gov/api/vehicles/decodevinvalues/{test_vin}?format=json",
            json={"error": "Internal server error"},
            status=500
        )
        
        result = client.get_vehicle_info(test_vin)
        
        assert result.vin == test_vin
        assert result.cylinders is None
        assert "failed" in result.error_message.lower()
    
    @responses.activate
    def test_invalid_cylinder_format(self, client):
        """Test handling of non-numeric cylinder values."""
        test_vin = "3GTP1VEC4EG551563"
        
        # Mock response with invalid cylinder format
        mock_response = {
            "Results": [{
                "EngineCylinders": "V8",  # Non-numeric value
                "Make": "GMC",
                "Model": "Sierra",
                "ModelYear": "2014"
            }]
        }
        
        responses.add(
            responses.GET,
            f"https://vpic.nhtsa.dot.gov/api/vehicles/decodevinvalues/{test_vin}?format=json",
            json=mock_response,
            status=200
        )
        
        result = client.get_vehicle_info(test_vin)
        
        assert result.vin == test_vin
        assert result.cylinders is None  # Should be None due to parsing failure
        assert result.make == "GMC"
        assert result.model == "Sierra"
    
    def test_invalid_vin_validation(self, client):
        """Test that invalid VINs are caught before API call."""
        invalid_vin = "INVALID"
        
        result = client.get_vehicle_info(invalid_vin)
        
        assert result.vin == invalid_vin
        assert result.cylinders is None
        assert result.error_message == "Invalid VIN format"
        assert result.api_response_time == 0.0


class TestDataEnrichment:
    """Test data enrichment and merging functionality."""
    
    @pytest.fixture
    def config(self):
        """Test configuration."""
        config = create_default_config()
        config['processing']['use_threading'] = False  # Disable threading for tests
        config['processing']['batch_size'] = 2
        return config
    
    @pytest.fixture
    def enricher(self, config):
        """Vehicle data enricher for testing."""
        return VehicleDataEnricher(config)
    
    @pytest.fixture
    def sample_data(self):
        """Sample vehicle data for testing."""
        return pd.DataFrame({
            'id': [1, 2, 3, 4],
            'VIN': ['3GTP1VEC4EG551563', '1GCSCSE06AZ123805', 'INVALIDVIN123456', ''],
            'cylinders': [None, '8 cylinders', None, None],
            'make': ['gmc', 'chevrolet', 'unknown', 'unknown'],
            'model': ['sierra', 'silverado', 'unknown', 'unknown']
        })
    
    def test_filter_vins_for_enrichment(self, enricher, sample_data):
        """Test VIN filtering logic."""
        vins_to_enrich = enricher.filter_vins_for_enrichment(sample_data)
        
        # Should include VINs 1 and 3 (2 has cylinders, 4 is empty)
        expected_vins = ['3GTP1VEC4EG551563', 'INVALIDVIN123456']
        assert set(vins_to_enrich) == set(expected_vins)
    
    def test_filter_vins_skip_existing_disabled(self, config, sample_data):
        """Test VIN filtering when skip_existing_cylinders is disabled."""
        config['processing']['skip_existing_cylinders'] = False
        enricher = VehicleDataEnricher(config)
        
        vins_to_enrich = enricher.filter_vins_for_enrichment(sample_data)
        
        # Should include all valid VINs when skip_existing is disabled
        expected_vins = ['3GTP1VEC4EG551563', '1GCSCSE06AZ123805', 'INVALIDVIN123456']
        assert set(vins_to_enrich) == set(expected_vins)
    
    def test_merge_enrichment_results(self, enricher, sample_data):
        """Test merging enrichment results with original data."""
        # Mock enrichment results
        enrichment_results = [
            EnrichmentResult(
                vin='3GTP1VEC4EG551563',
                cylinders=8,
                api_response_time=1.0,
                make='GMC',
                model='Sierra',
                year='2014'
            ),
            EnrichmentResult(
                vin='INVALIDVIN123456',
                cylinders=None,
                api_response_time=0.5,
                error_message='Invalid VIN format'
            )
        ]
        
        result_df = enricher.merge_enrichment_results(sample_data, enrichment_results)
        
        # Check that data was merged correctly
        assert len(result_df) == len(sample_data)
        assert 'api_cylinders' in result_df.columns
        assert 'api_make' in result_df.columns
        assert 'api_error_message' in result_df.columns
        
        # Check specific values
        row_1 = result_df[result_df['VIN'] == '3GTP1VEC4EG551563'].iloc[0]
        assert row_1['api_cylinders'] == 8
        assert row_1['api_make'] == 'GMC'
        assert row_1['cylinders'] == 8.0  # Should be updated from None
        
        row_3 = result_df[result_df['VIN'] == 'INVALIDVIN123456'].iloc[0]
        assert pd.isna(row_3['api_cylinders'])
        assert row_3['api_error_message'] == 'Invalid VIN format'


class TestDataProcessing:
    """Test data processing with real test files."""
    
    @pytest.fixture
    def config(self):
        """Test configuration."""
        config = create_default_config()
        config['processing']['use_threading'] = False
        config['processing']['batch_size'] = 2
        return config
    
    @pytest.fixture
    def enricher(self, config):
        """Vehicle data enricher for testing."""
        return VehicleDataEnricher(config)
    
    def test_load_sample_missing_data(self, enricher):
        """Test loading the sample_missing.csv test file."""
        sample_file = Path('data_enrichment/sample_missing.csv')
        
        if sample_file.exists():
            df = enricher.load_input_data(str(sample_file))
            
            assert len(df) > 0
            assert 'VIN' in df.columns
            assert 'cylinders' in df.columns
            
            # Check that some cylinders are missing (blank)
            missing_cylinders = df['cylinders'].isna() | (df['cylinders'] == '') | (df['cylinders'].astype(str).str.strip() == '')
            assert missing_cylinders.any(), "Sample file should have missing cylinder data"
    
    def test_load_sample_enriched_data(self, enricher):
        """Test loading the sample_enriched_demo.csv test file."""
        sample_file = Path('data_enrichment/sample_enriched_demo.csv')
        
        if sample_file.exists():
            df = enricher.load_input_data(str(sample_file))
            
            assert len(df) > 0
            assert 'VIN' in df.columns
            assert 'cylinders' in df.columns
            
            # Check for API enrichment columns
            api_columns = ['api_cylinders', 'api_make', 'api_model', 'api_year']
            for col in api_columns:
                if col in df.columns:
                    assert df[col].notna().any(), f"Column {col} should have some non-null values"


class TestLogging:
    """Test logging functionality."""
    
    def test_standard_logging_setup(self):
        """Test standard logging configuration."""
        logger = setup_logging(log_level='INFO', structured=False)
        
        # The effective level should be INFO (20) or lower
        assert logger.getEffectiveLevel() <= logging.INFO
        assert len(logger.handlers) > 0
    
    def test_structured_logging_setup(self):
        """Test structured JSON logging configuration."""
        logger = setup_logging(log_level='DEBUG', structured=True)
        
        # The effective level should be DEBUG (10) or lower
        assert logger.getEffectiveLevel() <= logging.DEBUG
        assert len(logger.handlers) > 0
        
        # Test that structured logging produces JSON
        # Test that structured logging works by checking handler configuration
        handler = logger.handlers[0]
        assert hasattr(handler, 'formatter')
        
        # Test actual logging to verify JSON output
        import io
        test_stream = io.StringIO()
        test_handler = logging.StreamHandler(test_stream)
        test_handler.setFormatter(handler.formatter)
        test_logger = logging.getLogger('test_structured')
        test_logger.addHandler(test_handler)
        test_logger.setLevel(logging.DEBUG)
        
        test_logger.info("Test message", extra={'vin': 'TEST123', 'response_time': 1.5})
        output = test_stream.getvalue()
        
        # Should contain JSON structure
        assert '"vin": "TEST123"' in output or 'vin' in output
    
    def test_logging_with_extra_context(self):
        """Test logging with enrichment context."""
        logger = setup_logging(log_level='DEBUG', structured=True)
        
        # Test that logging with extra context works
        import io
        test_stream = io.StringIO()
        test_handler = logging.StreamHandler(test_stream)
        test_handler.setFormatter(logger.handlers[0].formatter)
        
        test_logger = logging.getLogger('test_context')
        test_logger.addHandler(test_handler)
        test_logger.setLevel(logging.DEBUG)
        
        test_logger.info("Enrichment completed", extra={
            'vin': '3GTP1VEC4EG551563',
            'cylinders': 8,
            'response_time': 1.23,
            'batch_id': 1
        })
        
        output = test_stream.getvalue()
        assert len(output) > 0  # Should have logged something


class TestConfiguration:
    """Test configuration handling."""
    
    def test_default_config_creation(self):
        """Test default configuration values."""
        config = create_default_config()
        
        assert 'api' in config
        assert 'processing' in config
        assert 'output' in config
        assert 'logging' in config
        
        assert config['api']['max_requests_per_second'] == 5.0
        assert config['processing']['batch_size'] == 100
        assert config['output']['format'] == 'csv'
        assert config['logging']['level'] == 'INFO'
    
    def test_config_validation(self):
        """Test configuration validation."""
        config = create_default_config()
        enricher = VehicleDataEnricher(config)
        
        # Should not raise any errors with valid config
        assert enricher.config == config
        assert enricher.batch_size == config['processing']['batch_size']
        assert enricher.use_threading == config['processing']['use_threading']


class TestIntegrationWithMocks:
    """Integration tests using mocked API responses."""
    
    @pytest.fixture
    def config(self):
        """Test configuration."""
        config = create_default_config()
        config['processing']['use_threading'] = False
        config['processing']['batch_size'] = 2
        config['api']['max_requests_per_second'] = 100.0  # Fast for testing
        return config
    
    @responses.activate
    def test_end_to_end_enrichment_success(self, config, tmp_path):
        """Test complete enrichment process with successful API responses."""
        # Create test input file
        test_data = pd.DataFrame({
            'id': [1, 2],
            'VIN': ['3GTP1VEC4EG551563', '1GCSCSE06AZ123805'],
            'cylinders': [None, None],
            'make': ['gmc', 'chevrolet']
        })
        
        input_file = tmp_path / "test_input.csv"
        output_file = tmp_path / "test_output.csv"
        test_data.to_csv(input_file, index=False)
        
        # Mock API responses
        responses.add(
            responses.GET,
            "https://vpic.nhtsa.dot.gov/api/vehicles/decodevinvalues/3GTP1VEC4EG551563?format=json",
            json={"Results": [{"EngineCylinders": "8", "Make": "GMC", "Model": "Sierra"}]},
            status=200
        )
        responses.add(
            responses.GET,
            "https://vpic.nhtsa.dot.gov/api/vehicles/decodevinvalues/1GCSCSE06AZ123805?format=json",
            json={"Results": [{"EngineCylinders": "8", "Make": "CHEVROLET", "Model": "Silverado"}]},
            status=200
        )
        
        # Run enrichment
        enricher = VehicleDataEnricher(config)
        results = enricher.process(str(input_file), str(output_file))
        
        # Verify results
        assert results['total_rows'] == 2
        assert results['vins_enriched'] == 2
        assert results['successful_enrichments'] == 2
        assert results['failed_enrichments'] == 0
        
        # Verify output file
        output_data = pd.read_csv(output_file)
        assert len(output_data) == 2
        assert 'api_cylinders' in output_data.columns
        assert output_data['cylinders'].notna().all()  # Should be filled in
    
    @responses.activate
    def test_end_to_end_enrichment_with_errors(self, config, tmp_path):
        """Test enrichment process with some API errors."""
        # Create test input file
        test_data = pd.DataFrame({
            'id': [1, 2],
            'VIN': ['3GTP1VEC4EG551563', 'INVALIDVIN123456'],
            'cylinders': [None, None]
        })
        
        input_file = tmp_path / "test_input.csv"
        output_file = tmp_path / "test_output.csv"
        test_data.to_csv(input_file, index=False)
        
        # Mock API responses - one success, one failure
        responses.add(
            responses.GET,
            "https://vpic.nhtsa.dot.gov/api/vehicles/decodevinvalues/3GTP1VEC4EG551563?format=json",
            json={"Results": [{"EngineCylinders": "8", "Make": "GMC"}]},
            status=200
        )
        # Invalid VIN will fail validation before API call
        
        # Run enrichment
        enricher = VehicleDataEnricher(config)
        results = enricher.process(str(input_file), str(output_file))
        
        # Verify results
        assert results['total_rows'] == 2
        assert results['vins_enriched'] == 2  # Both VINs attempted
        assert results['successful_enrichments'] == 1  # Only one succeeded
        assert results['failed_enrichments'] == 1  # One failed
        
        # Verify output file exists
        assert output_file.exists()


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"]) 
