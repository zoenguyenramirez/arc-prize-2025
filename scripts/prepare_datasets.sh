#!/bin/bash

# Function to prepare ARC 2024 data
prepare_arc_2024() {
    echo "Preparing ARC 2024 data..."
    python -m src.prepare_data \
        --data-sources 2024_arc-agi_training 2024_arc-agi_evaluation \
        --output-file intermediate_data/prepared_dataset_arc_2024.pth
}

# Function to prepare ARC 2025 data
prepare_arc_2025() {
    echo "Preparing ARC 2025 data..."
    python -m src.prepare_data \
        --data-sources arc-agi_training arc-agi_evaluation \
        --output-file intermediate_data/prepared_dataset_arc_2025.pth
}

# Function to prepare BARC data
prepare_barc() {
    echo "Preparing BARC data..."
    python -m src.prepare_data \
        --jsonl-file /home/nikola/Code/GenII/200k_HEAVY_gpt4o-description-gpt4omini-code_generated_problems/data_100k.jsonl \
        --output-file intermediate_data/prepared_dataset_barc.pth
}

# Function to prepare RE-ARC data
prepare_rearc() {
    echo "Preparing RE-ARC data..."
    python -m src.prepare_data \
        --data-array /home/nikola/Code/GenII/re-arc/44_5000 \
        --output-file intermediate_data/prepared_dataset_rearc.pth
}

# Function to prepare synthesized data
prepare_synth() {
    echo "Preparing synthesized data..."
    python -m src.prepare_data \
        --data-sources synth_conditional_logic synth_array_indexing synth_arithmetic_operations \
                      synth_modular_arithmetic synth_find_min_max synth_count_occurrences \
                      synth_element_wise_operations synth_array_manipulation pattern \
        --output-file intermediate_data/prepared_dataset_synth.pth
}



# Function to print usage
print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --arc-2024    Prepare ARC 2024 data"
    echo "  --arc-2025    Prepare ARC 2025 data"
    echo "  --barc        Prepare BARC data"
    echo "  --rearc       Prepare RE-ARC data"
    echo "  --synth       Prepare synthesized data"
    echo ""
    echo "Examples:"
    echo "  $0 --arc-2024             # Prepare ARC 2024 data"
    echo "  $0 --arc-2024 --arc-2025  # Prepare both ARC datasets"
    echo "  $0 --barc --rearc         # Prepare BARC and RE-ARC data"
}

# Main execution
main() {
    local do_arc_2024=false
    local do_arc_2025=false
    local do_barc=false
    local do_rearc=false
    local do_synth=false
    local any_data=false

    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --arc-2024)
                do_arc_2024=true
                any_data=true
                shift
                ;;
            --arc-2025)
                do_arc_2025=true
                any_data=true
                shift
                ;;
            --barc)
                do_barc=true
                any_data=true
                shift
                ;;
            --rearc)
                do_rearc=true
                any_data=true
                shift
                ;;
            --synth)
                do_synth=true
                any_data=true
                shift
                ;;
            --help|-h)
                print_usage
                exit 0
                ;;
            *)
                echo "Unknown option: $1"
                print_usage
                exit 1
                ;;
        esac
    done

    # If no data options specified, show usage
    if [ "$any_data" = false ]; then
        print_usage
        exit 1
    fi

    # Source GPU memory check if available
    if [ -f "$(dirname "$0")/utils/check_gpu_memory.sh" ]; then
        source "$(dirname "$0")/utils/check_gpu_memory.sh"
    fi

    # Prepare requested data
    if [ "$do_arc_2024" = true ]; then
        prepare_arc_2024
    fi

    if [ "$do_arc_2025" = true ]; then
        prepare_arc_2025
    fi

    if [ "$do_barc" = true ]; then
        prepare_barc
    fi

    if [ "$do_rearc" = true ]; then
        prepare_rearc
    fi

    if [ "$do_synth" = true ]; then
        prepare_synth
    fi

    # Show generated files
    echo ""
    echo "Generated dataset files:"
    ls -lh intermediate_data/prepared_dataset_*.pth 2>/dev/null || echo "No dataset files found"
}

# Run the main function with command line arguments
main "$@"