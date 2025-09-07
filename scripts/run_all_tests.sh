#!/bin/bash
# Run all tests: unit, integration, and scripts

echo "======================================================================="
echo "RUNNING ALL TESTS"
echo "======================================================================="
echo ""

# Run unit tests
echo "1. UNIT TESTS"
echo "----------------------------------------------------------------------"
python -m pytest tests/unit/ -v --tb=short

echo ""
echo "2. INTEGRATION TESTS"
echo "----------------------------------------------------------------------"
python -m pytest tests/integration/ -v --tb=short

echo ""
echo "3. SCRIPT TESTS"
echo "----------------------------------------------------------------------"
# Run shell script tests if they exist
if [ -f "tests/scripts/test_pseudo_rl_complete.sh" ]; then
    echo "Running pseudo-RL complete test..."
    bash tests/scripts/test_pseudo_rl_complete.sh
fi

echo ""
echo "======================================================================="
echo "ALL TESTS COMPLETE"
echo "======================================================================="