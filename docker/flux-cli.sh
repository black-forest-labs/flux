#!/bin/bash

# Enhanced CLI wrapper for FLUX Docker implementation
# Supports both local model inference and API access

set -e

# Default values
USE_LOCAL=false
API_KEY="${BFL_API_KEY:-}"
MODEL="flux.1-pro"
PROMPT=""
OUTPUT="flux-output.jpg"
OUTPUT_FORMAT="save"
GPU_SUPPORT=""

# Set up working directories
WORK_DIR="$(pwd)"
FLUX_HOME="${FLUX_HOME:-$HOME/.flux}"
FLUX_OUTPUTS="${FLUX_OUTPUTS:-$FLUX_HOME/outputs}"
FLUX_MODELS="${FLUX_MODELS:-$FLUX_HOME/models}"

# Ensure directories exist
mkdir -p "$FLUX_OUTPUTS"
mkdir -p "$FLUX_MODELS"

usage() {
    cat << EOF
Usage: $0 [options]

Options:
    --local             Use local model instead of API
    --api-key KEY       API key for remote usage
    --model NAME        Model name to use (default: flux.1-pro)
    --prompt TEXT       Prompt for image generation
    --output PATH       Output path (default: flux-output.jpg)
    --format FORMAT     Output format: save|url|image (default: save)
    --gpu              Enable GPU support
    -h, --help         Show this help message

Environment variables:
    FLUX_HOME          Base directory for FLUX data (default: ~/.flux)
    FLUX_OUTPUTS       Output directory (default: $FLUX_HOME/outputs)
    FLUX_MODELS        Models directory (default: $FLUX_HOME/models)
    BFL_API_KEY        API key (can be set instead of --api-key)

Examples:
    $0 --prompt "A beautiful sunset" --output sunset.jpg
    $0 --local --model flux.1-schnell --prompt "A forest" --gpu
EOF
}

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --local) USE_LOCAL=true ;;
        --api-key) API_KEY="$2"; shift ;;
        --model) MODEL="$2"; shift ;;
        --prompt) PROMPT="$2"; shift ;;
        --output) OUTPUT="$2"; shift ;;
        --format) OUTPUT_FORMAT="$2"; shift ;;
        --gpu) GPU_SUPPORT="--gpus all" ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown parameter: $1"; usage; exit 1 ;;
    esac
    shift
done

# Validate required arguments
if [ -z "$PROMPT" ]; then
    echo "Error: --prompt is required"
    usage
    exit 1
fi

if [ "$USE_LOCAL" = true ] && [ -z "$MODEL" ]; then
    echo "Error: --model is required when using local mode"
    usage
    exit 1
fi

if [ "$USE_LOCAL" = false ] && [ -z "$API_KEY" ]; then
    echo "Error: --api-key is required when using API mode"
    usage
    exit 1
fi

# Handle output path
if [[ "$OUTPUT" = /* ]]; then
    # Absolute path
    FINAL_OUTPUT="$OUTPUT"
    OUTPUT_DIR="$(dirname "$OUTPUT")"
else
    # Relative path - make it relative to current directory
    FINAL_OUTPUT="$WORK_DIR/$OUTPUT"
    OUTPUT_DIR="$(dirname "$FINAL_OUTPUT")"
fi

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

# Build Docker command
DOCKER_CMD="docker run --rm ${GPU_SUPPORT} \
    -v $FLUX_OUTPUTS:/app/outputs \
    -v $FLUX_MODELS:/app/models \
    -v $OUTPUT_DIR:/app/current"

if [ "$USE_LOCAL" = false ]; then
    DOCKER_CMD="$DOCKER_CMD -e BFL_API_KEY=$API_KEY"
fi

# Execute Docker command
if [ "$USE_LOCAL" = true ]; then
    $DOCKER_CMD flux-project \
        --model "$MODEL" \
        --prompt "$PROMPT" \
        "$OUTPUT_FORMAT" "/app/current/$(basename "$OUTPUT")"
else
    $DOCKER_CMD flux-project \
        --prompt "$PROMPT" \
        "$OUTPUT_FORMAT" "/app/current/$(basename "$OUTPUT")"
fi

echo "Output saved to: $FINAL_OUTPUT"
