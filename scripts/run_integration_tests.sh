#!/bin/bash
# Run integration tests

echo "======================================================================="
echo "RUNNING INTEGRATION TESTS"
echo "======================================================================="
echo ""

# Run Python integration tests
echo "Running integration tests from tests/integration/..."
python -m pytest tests/integration/ -v --tb=short

echo ""
echo "======================================================================="
echo "INTEGRATION TESTS COMPLETE"
echo "======================================================================="