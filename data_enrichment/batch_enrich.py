#!/usr/bin/env python3
"""
Batch Vehicle Data Enrichment Script

This script provides advanced batch processing capabilities for enriching large vehicle datasets
with cylinder count information from the NHTSA Vehicle API.

Features:
- Resume capability from checkpoint files
- Memory-efficient chunk processing
- Detailed progress tracking and statistics
- Error recovery and retry logic
- Support for distributed processing

Usage:
    python batch_enrich.py --input data/vehicles_cleaned.csv --output data/vehicles_enriched.csv
    python batch_enrich.py --config config.yaml --resume
"""

import pandas as pd
import json
import time
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Set
from dataclasses import asdict
import pickle
import hashlib

from enrich_with_cylinders import (
    VehicleDataEnricher, 
    load_config, 
    create_default_config,
    setup_logging,
    EnrichmentResult
)


class BatchEnrichmentManager:
    """Advanced batch enrichment manager with checkpointing and resume capabilities."""
    
    def __init__(self, config: Dict[str, Any], checkpoint_dir: str = "checkpoints"):
        self.config = config
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        # Initialize enricher
        self.enricher = VehicleDataEnricher(config)
        
        # Batch processing configuration
        self.chunk_size = config.get('batch', {}).get('chunk_size', 10000)
        self.save_frequency = config.get('batch', {}).get('save_frequency', 5)  # Save every N chunks
        
        # State tracking
        self.processed_vins: Set[str] = set()
        self.enrichment_results: List[EnrichmentResult] = []
        self.total_chunks = 0
        self.processed_chunks = 0
    
    def generate_checkpoint_id(self, input_path: str, config: Dict[str, Any]) -> str:
        """Generate a unique checkpoint ID based on input file and config."""
        # Create a hash of the input file path and relevant config parameters
        config_subset = {
            'api': config.get('api', {}),
            'processing': config.get('processing', {}),
            'skip_existing_cylinders': config.get('processing', {}).get('skip_existing_cylinders', True)
        }
        
        hash_input = f"{input_path}:{json.dumps(config_subset, sort_keys=True)}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:12]
    
    def save_checkpoint(self, checkpoint_id: str, chunk_num: int, 
                       processed_vins: Set[str], results: List[EnrichmentResult]):
        """Save processing checkpoint."""
        checkpoint_file = self.checkpoint_dir / f"checkpoint_{checkpoint_id}.pkl"
        
        checkpoint_data = {
            'chunk_num': chunk_num,
            'processed_vins': processed_vins,
            'results': [asdict(result) for result in results],
            'timestamp': time.time(),
            'config': self.config
        }
        
        try:
            with open(checkpoint_file, 'wb') as f:
                pickle.dump(checkpoint_data, f)
            
            self.logger.info(f"Checkpoint saved: {checkpoint_file}")
            
        except Exception as e:
            self.logger.warning(f"Failed to save checkpoint: {e}")
    
    def load_checkpoint(self, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        """Load processing checkpoint."""
        checkpoint_file = self.checkpoint_dir / f"checkpoint_{checkpoint_id}.pkl"
        
        if not checkpoint_file.exists():
            return None
        
        try:
            with open(checkpoint_file, 'rb') as f:
                checkpoint_data = pickle.load(f)
            
            self.logger.info(f"Checkpoint loaded: {checkpoint_file}")
            return checkpoint_data
            
        except Exception as e:
            self.logger.warning(f"Failed to load checkpoint: {e}")
            return None
    
    def get_unique_vins_from_chunk(self, chunk: pd.DataFrame) -> List[str]:
        """Extract unique VINs from a data chunk, excluding already processed ones."""
        if 'VIN' not in chunk.columns:
            return []
        
        # Filter valid VINs
        valid_vins = chunk[chunk['VIN'].notna() & (chunk['VIN'].str.strip() != '')]
        
        # Get unique VINs not already processed
        chunk_vins = set(valid_vins['VIN'].unique()) - self.processed_vins
        
        return list(chunk_vins)
    
    def process_file_in_chunks(self, input_path: str, output_path: str, 
                              resume: bool = False) -> Dict[str, Any]:
        """Process large files in chunks with checkpointing."""
        checkpoint_id = self.generate_checkpoint_id(input_path, self.config)
        start_chunk = 0
        
        # Try to resume from checkpoint
        if resume:
            checkpoint_data = self.load_checkpoint(checkpoint_id)
            if checkpoint_data:
                start_chunk = checkpoint_data['chunk_num']
                self.processed_vins = checkpoint_data['processed_vins']
                
                # Reconstruct EnrichmentResult objects
                self.enrichment_results = [
                    EnrichmentResult(**result_dict) 
                    for result_dict in checkpoint_data['results']
                ]
                
                self.logger.info(f"Resuming from chunk {start_chunk} with {len(self.processed_vins)} processed VINs")
            else:
                self.logger.info("No checkpoint found, starting from beginning")
        
        # Get file info
        file_size = Path(input_path).stat().st_size
        self.logger.info(f"Processing file: {input_path} ({file_size / (1024**3):.2f} GB)")
        
        # Read file in chunks
        chunk_reader = pd.read_csv(input_path, chunksize=self.chunk_size)
        
        # Estimate total chunks for progress tracking
        total_rows = sum(1 for _ in open(input_path)) - 1  # Subtract header
        self.total_chunks = (total_rows + self.chunk_size - 1) // self.chunk_size
        
        self.logger.info(f"Processing {total_rows:,} rows in ~{self.total_chunks} chunks")
        
        start_time = time.time()
        original_df_chunks = []  # Store original data for final merge
        
        try:
            for chunk_num, chunk in enumerate(chunk_reader):
                # Skip chunks if resuming
                if chunk_num < start_chunk:
                    original_df_chunks.append(chunk)
                    continue
                
                self.processed_chunks = chunk_num + 1
                
                self.logger.info(f"Processing chunk {self.processed_chunks}/{self.total_chunks} "
                               f"({len(chunk):,} rows)")
                
                # Store original chunk for final merge
                original_df_chunks.append(chunk)
                
                # Get VINs to enrich from this chunk
                chunk_vins = self.get_unique_vins_from_chunk(chunk)
                
                if chunk_vins:
                    self.logger.info(f"Enriching {len(chunk_vins)} unique VINs from chunk")
                    
                    # Enrich VINs
                    chunk_results = self.enricher.enrich_vins(chunk_vins)
                    self.enrichment_results.extend(chunk_results)
                    
                    # Update processed VINs set
                    self.processed_vins.update(chunk_vins)
                    
                    # Log chunk statistics
                    successful = sum(1 for r in chunk_results if r.cylinders is not None)
                    self.logger.info(f"Chunk {self.processed_chunks}: {successful}/{len(chunk_results)} successful")
                else:
                    self.logger.info(f"No new VINs to enrich in chunk {self.processed_chunks}")
                
                # Save checkpoint periodically
                if self.processed_chunks % self.save_frequency == 0:
                    self.save_checkpoint(checkpoint_id, self.processed_chunks, 
                                       self.processed_vins, self.enrichment_results)
                
                # Log overall progress
                elapsed = time.time() - start_time
                chunks_remaining = self.total_chunks - self.processed_chunks
                if self.processed_chunks > start_chunk:
                    avg_time_per_chunk = elapsed / (self.processed_chunks - start_chunk)
                    eta = avg_time_per_chunk * chunks_remaining
                    self.logger.info(f"Progress: {self.processed_chunks}/{self.total_chunks} "
                                   f"({(self.processed_chunks/self.total_chunks)*100:.1f}%) "
                                   f"ETA: {eta/60:.1f} minutes")
        
        except KeyboardInterrupt:
            self.logger.warning("Processing interrupted by user")
            # Save checkpoint before exiting
            self.save_checkpoint(checkpoint_id, self.processed_chunks, 
                               self.processed_vins, self.enrichment_results)
            raise
        
        except Exception as e:
            self.logger.error(f"Error during processing: {e}")
            # Save checkpoint before re-raising
            self.save_checkpoint(checkpoint_id, self.processed_chunks, 
                               self.processed_vins, self.enrichment_results)
            raise
        
        # Combine all chunks back into full dataframe
        self.logger.info("Combining chunks and merging enrichment results...")
        full_df = pd.concat(original_df_chunks, ignore_index=True)
        
        # Merge enrichment results
        enriched_df = self.enricher.merge_enrichment_results(full_df, self.enrichment_results)
        
        # Save final result
        self.enricher.save_enriched_data(enriched_df, output_path)
        
        # Clean up checkpoint
        checkpoint_file = self.checkpoint_dir / f"checkpoint_{checkpoint_id}.pkl"
        if checkpoint_file.exists():
            checkpoint_file.unlink()
            self.logger.info("Checkpoint file cleaned up")
        
        # Calculate final statistics
        total_time = time.time() - start_time
        successful_enrichments = sum(1 for r in self.enrichment_results if r.cylinders is not None)
        
        stats = {
            'total_rows': len(full_df),
            'unique_vins_processed': len(self.processed_vins),
            'successful_enrichments': successful_enrichments,
            'failed_enrichments': len(self.enrichment_results) - successful_enrichments,
            'processing_time': total_time,
            'chunks_processed': self.processed_chunks
        }
        
        self.logger.info("=" * 60)
        self.logger.info("BATCH ENRICHMENT SUMMARY")
        self.logger.info("=" * 60)
        self.logger.info(f"Total rows: {stats['total_rows']:,}")
        self.logger.info(f"Unique VINs processed: {stats['unique_vins_processed']:,}")
        self.logger.info(f"Successful enrichments: {stats['successful_enrichments']:,}")
        self.logger.info(f"Failed enrichments: {stats['failed_enrichments']:,}")
        self.logger.info(f"Chunks processed: {stats['chunks_processed']}")
        self.logger.info(f"Total time: {stats['processing_time']/60:.1f} minutes")
        if stats['unique_vins_processed'] > 0:
            success_rate = (stats['successful_enrichments'] / stats['unique_vins_processed']) * 100
            self.logger.info(f"Success rate: {success_rate:.1f}%")
        
        return stats


def main():
    """Main entry point for batch enrichment."""
    parser = argparse.ArgumentParser(
        description="Batch vehicle data enrichment with checkpointing",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--input', '-i', type=str, required=True,
                       help='Input vehicle data file (CSV)')
    parser.add_argument('--output', '-o', type=str, required=True,
                       help='Output enriched data file')
    parser.add_argument('--config', '-c', type=str,
                       help='Configuration file (YAML)')
    parser.add_argument('--chunk-size', type=int, default=10000,
                       help='Number of rows per chunk')
    parser.add_argument('--save-frequency', type=int, default=5,
                       help='Save checkpoint every N chunks')
    parser.add_argument('--checkpoint-dir', type=str, default="checkpoints",
                       help='Directory for checkpoint files')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from previous checkpoint')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help='Logging level')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        config = load_config(args.config)
    else:
        config = create_default_config()
    
    # Override config with command-line arguments
    if not config.get('batch'):
        config['batch'] = {}
    
    config['batch']['chunk_size'] = args.chunk_size
    config['batch']['save_frequency'] = args.save_frequency
    
    # Set up logging
    logger = setup_logging(log_level=args.log_level)
    
    logger.info("Starting batch vehicle data enrichment")
    logger.info(f"Input: {args.input}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Chunk size: {args.chunk_size:,}")
    logger.info(f"Resume mode: {args.resume}")
    
    try:
        # Initialize batch manager and process
        batch_manager = BatchEnrichmentManager(config, args.checkpoint_dir)
        results = batch_manager.process_file_in_chunks(
            args.input, 
            args.output, 
            resume=args.resume
        )
        
        logger.info("Batch enrichment completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Batch enrichment failed: {e}")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main()) 
