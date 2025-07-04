"""
NHTSA Vehicle Data Enrichment Pipeline

This package provides tools for enriching vehicle data with NHTSA API information,
particularly for adding missing cylinder data to vehicle datasets.

Main modules:
- enrich_with_cylinders: Core enrichment functionality
- batch_enrich: Batch processing utilities
"""

from .enrich_with_cylinders import VehicleDataEnricher, create_default_config
from .batch_enrich import BatchEnrichmentManager

__version__ = "1.0.0"
__all__ = ["VehicleDataEnricher", "create_default_config", "BatchEnrichmentManager"] 
