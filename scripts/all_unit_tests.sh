#!/bin/bash
# Run all unit tests (not integration tests)

echo "======================================================================="
echo "RUNNING UNIT TESTS"
echo "======================================================================="
echo ""

# Run unit tests with both naming conventions
echo "Running unit tests from tests/unit/..."
python -m pytest tests/unit/ -v --tb=short

# Alternative: Use unittest discovery (original method)
# python -m unittest discover tests/unit -p "*_test.py"
# python -m unittest discover tests/unit -p "test_*.py"

echo ""
echo "======================================================================="
echo "UNIT TESTS COMPLETE"
echo "======================================================================="
echo ""
echo "To run integration tests: python -m pytest tests/integration/"
echo "To run all tests: python -m pytest tests/"