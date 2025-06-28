#!/usr/bin/env python3
"""
Test Runner for Data Enrichment Pipeline

This script provides convenient commands for running different types of tests:
- Unit tests (fast, no external dependencies)
- Integration tests (with mocked API calls)
- Coverage reports
- Smoke tests for quick validation

Usage:
    python run_tests.py --unit          # Run unit tests only
    python run_tests.py --integration   # Run integration tests
    python run_tests.py --all           # Run all tests
    python run_tests.py --coverage      # Run with coverage report
    python run_tests.py --smoke         # Quick smoke test
"""

import subprocess
import sys
import argparse
from pathlib import Path


def run_command(cmd: list, description: str):
    """Run a command and handle errors."""
    print(f"\n🔄 {description}")
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Success!")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed with exit code {e.returncode}")
        if e.stdout:
            print("STDOUT:", e.stdout)
        if e.stderr:
            print("STDERR:", e.stderr)
        return False
    except FileNotFoundError:
        print(f"❌ Command not found: {cmd[0]}")
        print("Make sure pytest is installed: pip install pytest responses pytest-cov")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test runner for data enrichment pipeline")
    parser.add_argument('--unit', action='store_true', help='Run unit tests only')
    parser.add_argument('--integration', action='store_true', help='Run integration tests')
    parser.add_argument('--all', action='store_true', help='Run all tests')
    parser.add_argument('--coverage', action='store_true', help='Run with coverage report')
    parser.add_argument('--smoke', action='store_true', help='Run quick smoke test')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Default to running all tests if no specific option is given
    if not any([args.unit, args.integration, args.all, args.coverage, args.smoke]):
        args.all = True
    
    success = True
    test_dir = Path(__file__).parent
    
    # Change to test directory
    import os
    original_dir = os.getcwd()
    os.chdir(test_dir)
    
    try:
        if args.smoke:
            # Quick smoke test - just VIN validation and basic functionality
            cmd = ['python', '-m', 'pytest', 'test_enrichment.py::TestVINValidation', '-v']
            success &= run_command(cmd, "Running smoke tests")
        
        elif args.unit:
            # Unit tests - fast tests with no external dependencies
            cmd = ['python', '-m', 'pytest', '-m', 'unit', '-v']
            if args.verbose:
                cmd.append('-s')
            success &= run_command(cmd, "Running unit tests")
        
        elif args.integration:
            # Integration tests - tests with mocked external dependencies
            cmd = ['python', '-m', 'pytest', '-m', 'integration', '-v']
            if args.verbose:
                cmd.append('-s')
            success &= run_command(cmd, "Running integration tests")
        
        elif args.coverage:
            # Run all tests with coverage
            cmd = ['python', '-m', 'pytest', 'test_enrichment.py', 
                   '--cov=enrich_with_cylinders', '--cov-report=term-missing', 
                   '--cov-report=html:htmlcov', '-v']
            if args.verbose:
                cmd.append('-s')
            success &= run_command(cmd, "Running tests with coverage")
            
            print("\n📊 Coverage report generated in htmlcov/index.html")
        
        elif args.all:
            # Run all tests
            cmd = ['python', '-m', 'pytest', 'test_enrichment.py', '-v']
            if args.verbose:
                cmd.append('-s')
            success &= run_command(cmd, "Running all tests")
        
        # Test the actual API functionality (optional, requires network)
        if success and (args.all or args.integration):
            print("\n🌐 Testing live API connectivity (optional)...")
            try:
                cmd = ['python', 'test_api.py', '--vin', '3GTP1VEC4EG551563']
                subprocess.run(cmd, check=True, timeout=30)
                print("✅ Live API test successful!")
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                print("⚠️  Live API test failed (this is okay - network/API may be unavailable)")
            except FileNotFoundError:
                print("⚠️  test_api.py not found - skipping live API test")
    
    finally:
        # Return to original directory
        os.chdir(original_dir)
    
    if success:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print("\n💥 Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 
