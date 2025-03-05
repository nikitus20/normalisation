#\!/bin/bash
# Script to download model weights for normalization experiments

# Create directory for downloaded weights
WEIGHTS_DIR="weights"
mkdir -p $WEIGHTS_DIR

# Helper function to download and verify
download_and_verify() {
    MODEL_NAME=$1
    MODEL_PATH="${WEIGHTS_DIR}/${MODEL_NAME}"
    
    echo "Downloading ${MODEL_NAME} model weights..."
    
    # Use huggingface-cli to download
    if \! command -v huggingface-cli &> /dev/null; then
        pip install huggingface_hub
    fi
    
    # Download model using HF CLI
    huggingface-cli download --resume-download --local-dir "${MODEL_PATH}" "${MODEL_NAME}"
    
    # Check if download succeeded
    if [ $? -eq 0 ]; then
        echo "✅ Successfully downloaded ${MODEL_NAME}"
    else
        echo "❌ Failed to download ${MODEL_NAME}"
    fi
}

# Download required model weights
echo "Downloading model weights for experiments"
echo "This may take some time depending on your internet connection..."

# Download LLaMA-7B tokenizer (needed for experiments)
download_and_verify "facebook/llama-7b"

echo "Weights download complete\!"
echo "You can now run the experiments using the downloaded weights."
