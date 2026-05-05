#!/bin/bash

# Quick Start Script for Bloque2_CNN Training & Evaluation
# Usage: bash quick_start.sh [command]

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
CONFIG_FILE="config.yaml"
TASK="${1:-all}"
MODEL="${2:-unet}"

echo -e "${GREEN}=== Bloque2_CNN Training & Evaluation Quick Start ===${NC}\n"

# Function: Check if Python is available
check_python() {
    if ! command -v python &> /dev/null; then
        echo -e "${RED}Error: Python not found. Please install Python 3.8+${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ Python found: $(python --version)${NC}\n"
}

# Function: Check if CUDA is available
check_cuda() {
    python -c "import torch; print(f'✓ PyTorch found: {torch.__version__}')"
    if python -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
        echo -e "${GREEN}✓ CUDA available${NC}\n"
    else
        echo -e "${YELLOW}⚠ CUDA not available - will use CPU (slow)${NC}\n"
    fi
}

# Function: Install dependencies
install_deps() {
    echo -e "${YELLOW}Installing dependencies...${NC}"
    pip install -r requirements.txt -q
    echo -e "${GREEN}✓ Dependencies installed${NC}\n"
}

# Function: Display help
show_help() {
    echo "Available commands:"
    echo "  bash quick_start.sh train [task] [model]     - Train models"
    echo "  bash quick_start.sh eval [task] [model]      - Evaluate models"
    echo "  bash quick_start.sh full [task]              - Train + Evaluate"
    echo "  bash quick_start.sh help                     - Show this message"
    echo ""
    echo "Tasks: all, classification, detection, segmentation"
    echo "Models (segmentation only): unet, deeplabv3plus"
    echo ""
    echo "Examples:"
    echo "  bash quick_start.sh train classification"
    echo "  bash quick_start.sh eval detection"
    echo "  bash quick_start.sh full all"
}

# Function: Train
train_models() {
    echo -e "${YELLOW}Starting training for task: ${TASK}${NC}\n"
    python -u -m src.train --config $CONFIG_FILE --task $TASK --model $MODEL 2>&1 | tee train_log.txt
    echo -e "${GREEN}✓ Training completed${NC}\n"
}

# Function: Evaluate
evaluate_models() {
    echo -e "${YELLOW}Starting evaluation for task: ${TASK}${NC}\n"
    python -u -m src.evaluate --config $CONFIG_FILE --task $TASK --seg-model $MODEL 2>&1 | tee eval_log.txt
    echo -e "${GREEN}✓ Evaluation completed${NC}\n"
}

# Function: Full pipeline
full_pipeline() {
    check_python
    check_cuda
    train_models
    evaluate_models
}

# Main logic
case "${1:-help}" in
    install)
        check_python
        install_deps
        ;;
    train)
        check_python
        check_cuda
        train_models
        ;;
    eval)
        check_python
        check_cuda
        evaluate_models
        ;;
    full)
        full_pipeline
        ;;
    setup)
        echo -e "${YELLOW}Checking environment...${NC}\n"
        check_python
        check_cuda
        echo -e "${GREEN}✓ Environment ready for training${NC}"
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        echo -e "${RED}Unknown command: $1${NC}\n"
        show_help
        exit 1
        ;;
esac

exit 0
