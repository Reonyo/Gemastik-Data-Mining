"""
Test Runner for Legal Knowledge Base Project

Run all tests in the test directory.
Usage:
    python test/run_tests.py
    python test/run_tests.py --verbose
    python test/run_tests.py --specific test_vector_indexing
"""

import sys
import logging
import argparse
from pathlib import Path
import importlib.util

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_test_module(test_file: Path):
    """
    Run a specific test module.
    
    Args:
        test_file: Path to the test module
        
    Returns:
        bool: True if test passed, False otherwise
    """
    try:
        # Import the test module
        spec = importlib.util.spec_from_file_location("test_module", test_file)
        test_module = importlib.util.module_from_spec(spec)
        
        # Execute the module
        spec.loader.exec_module(test_module)
        
        # Run main function if it exists
        if hasattr(test_module, 'main'):
            logger.info(f"Running test: {test_file.name}")
            result = test_module.main()
            return result
        else:
            logger.warning(f"No main() function found in {test_file.name}")
            return True
            
    except Exception as e:
        logger.error(f"Failed to run test {test_file.name}: {e}")
        return False


def discover_tests(test_dir: Path, pattern: str = "test_*.py"):
    """
    Discover test files in the test directory.
    
    Args:
        test_dir: Path to test directory
        pattern: File pattern to match
        
    Returns:
        list: List of test file paths
    """
    test_files = []
    
    for test_file in test_dir.glob(pattern):
        if test_file.name != "__init__.py" and test_file.name != "run_tests.py":
            test_files.append(test_file)
    
    return sorted(test_files)


def main():
    """Main test runner."""
    parser = argparse.ArgumentParser(description="Run tests for Legal Knowledge Base project")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    parser.add_argument("--specific", "-s", type=str, help="Run specific test module")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    test_dir = Path(__file__).parent
    
    if args.specific:
        # Run specific test
        test_file = test_dir / f"{args.specific}.py"
        if not test_file.exists():
            logger.error(f"Test file not found: {test_file}")
            return False
        
        test_files = [test_file]
    else:
        # Discover all tests
        test_files = discover_tests(test_dir)
    
    if not test_files:
        logger.warning("No test files found")
        return True
    
    logger.info(f"🚀 Running {len(test_files)} test module(s)")
    
    results = {}
    
    for test_file in test_files:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running test module: {test_file.name}")
        logger.info('='*60)
        
        result = run_test_module(test_file)
        results[test_file.name] = result
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("TEST RUNNER SUMMARY")
    logger.info('='*60)
    
    passed = 0
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
        if result:
            passed += 1
    
    logger.info(f"\nOverall: {passed}/{len(results)} test modules passed")
    
    if passed == len(results):
        logger.info("🎉 All tests passed!")
        return True
    else:
        logger.error("⚠️ Some tests failed.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
