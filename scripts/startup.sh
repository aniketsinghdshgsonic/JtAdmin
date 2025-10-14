









# #!/bin/bash
# # scripts/startup.sh
# # Streamlined startup script for production use

# set -euo pipefail

# # Color codes
# RED='\033[0;31m'
# GREEN='\033[0;32m'
# YELLOW='\033[1;33m'
# BLUE='\033[0;34m'
# NC='\033[0m'

# log() {
#     echo -e "${BLUE}[$(date +'%H:%M:%S')]${NC} $1"
# }

# log_success() {
#     echo -e "${GREEN}[$(date +'%H:%M:%S')]${NC} ✓ $1"
# }

# log_warning() {
#     echo -e "${YELLOW}[$(date +'%H:%M:%S')]${NC} ⚠ $1"
# }

# log_error() {
#     echo -e "${RED}[$(date +'%H:%M:%S')]${NC} ✗ $1"
# }

# command_exists() {
#     command -v "$1" >/dev/null 2>&1
# }

# # Simplified GPU check
# check_gpu_environment() {
#     if command_exists nvidia-smi && nvidia-smi > /dev/null 2>&1; then
#         log_success "GPU accessible"
        
#         # Quick CUDA test
#         python3 -c "
# import torch
# print(f'CUDA: {torch.cuda.is_available()}')
# if torch.cuda.is_available():
#     props = torch.cuda.get_device_properties(0)
#     print(f'GPU: {props.name} ({props.total_memory / 1024**3:.0f}GB)')
# " 2>/dev/null || {
#             log_error "CUDA not available in PyTorch"
#             return 1
#         }
        
#         return 0
#     else
#         log_error "GPU not accessible - ensure container started with --gpus all"
#         return 1
#     fi
# }

# # Quick model check
# check_model_files() {
#     HF_MODEL_PATH="/app/ai-model/CodeLlama-13b-hf"
#     GGUF_MODEL_PATH="/app/ai-model/codellama-13b.Q4_K_M.gguf"
    
#     if [ -d "$HF_MODEL_PATH" ]; then
#         log_success "HuggingFace model found"
#     else
#         log_warning "HuggingFace model missing - training disabled"
#     fi
    
#     if [ -f "$GGUF_MODEL_PATH" ]; then
#         log_success "GGUF model found"
#     else
#         log_warning "GGUF model missing - inference may be limited"
#     fi
# }

# # Essential imports test
# test_critical_imports() {
#     python3 -c "
# import sys
# sys.path.append('/app')

# try:
#     import torch, transformers, peft, bitsandbytes
#     from app.services.llm_training_manager import LLMTrainingManager
#     from app.models.llama_manager import LlamaManager
#     print('✓ All critical imports successful')
# except ImportError as e:
#     print(f'✗ Import failed: {e}')
#     sys.exit(1)
# " || {
#         log_error "Critical imports failed"
#         return 1
#     }
# }

# # Quick Milvus check
# check_milvus() {
#     local host="${MILVUS_HOST:-localhost}"
#     local port="${MILVUS_PORT:-19530}"
    
#     if command_exists nc && nc -z "$host" "$port" 2>/dev/null; then
#         log_success "Milvus available"
#         return 0
#     else
#         log_warning "Milvus not available - vector features disabled"
#         return 1
#     fi
# }

# echo "============================================"
# echo "🚀 LLM Training System Starting..."
# echo "============================================"

# # Setup environment
# cd /app
# export PYTHONPATH="/app:${PYTHONPATH:-}"
# mkdir -p logs .cache temp ai-model/trained_models ai-model/training_data

# # Core system checks
# log "Checking system environment..."

# # GPU check (critical for training)
# if ! check_gpu_environment; then
#     log_error "GPU issues detected - training functionality limited"
# fi

# # Model files check
# check_model_files

# # Python dependencies check
# if ! test_critical_imports; then
#     log_error "Python dependencies failed"
#     exit 1
# fi

# # Optional services
# check_milvus || true

# # Set final environment variables
# export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
# export TOKENIZERS_PARALLELISM="false"
# export TRANSFORMERS_CACHE="/app/.cache"

# # Final system info (concise)
# python3 -c "
# import psutil, torch
# print(f'System: {psutil.cpu_count()} cores, {psutil.virtual_memory().total / 1024**3:.0f}GB RAM')
# if torch.cuda.is_available():
#     props = torch.cuda.get_device_properties(0)
#     available_mem = (props.total_memory - torch.cuda.memory_allocated()) / 1024**3
#     print(f'GPU: {available_mem:.0f}GB available memory')
# " 2>/dev/null || echo "System info unavailable"

# # Start application
# if [ ! -f "main.py" ]; then
#     log_error "main.py not found"
#     exit 1
# fi

# log_success "System ready - starting application"
# exec python3 main.py














#!/bin/bash
# scripts/startup.sh - ENHANCED DEBUG VERSION
# set -euo pipefail
set -eo pipefail

# Debug mode from environment
DEBUG_STARTUP=${STARTUP_DEBUG:-false}

log() {
    echo "[$(date +'%H:%M:%S')] $1"
    if [ "$DEBUG_STARTUP" = "true" ]; then
        echo "[$(date +'%H:%M:%S')] $1" >> /app/logs/startup.log 2>/dev/null || true
    fi
}

log_error() {
    echo "ERROR [$(date +'%H:%M:%S')] $1" >&2
    if [ "$DEBUG_STARTUP" = "true" ]; then
        echo "ERROR [$(date +'%H:%M:%S')] $1" >> /app/logs/startup.log 2>/dev/null || true
    fi
}

# Enhanced error handler
handle_error() {
    local exit_code=$?
    local line_number=$1
    log_error "Script failed at line $line_number with exit code $exit_code"
    
    # Clean up restart signal file on error
    if [ -f "/app/temp/restart_requested.signal" ]; then
        log "Cleaning up restart signal due to startup failure"
        rm -f "/app/temp/restart_requested.signal" 2>/dev/null || true
    fi
    
    # Show environment for debugging
    if [ "$DEBUG_STARTUP" = "true" ]; then
        log_error "Environment at failure:"
        env | grep -E "(CUDA|PYTORCH|PYTHON)" || true
        log_error "Working directory: $(pwd)"
        log_error "Python path: $PYTHONPATH"
        log_error "Available disk space:"
        df -h /app || true
    fi
    
    exit $exit_code
}

trap 'handle_error $LINENO' ERR

echo "============================================"
echo "🚀 LLM Training System Starting (Debug: $DEBUG_STARTUP)..."
echo "============================================"

# Setup
cd /app
export PYTHONPATH="/app:${PYTHONPATH:-}"
mkdir -p logs .cache temp ai-model/trained_models ai-model/training_data

# Check for restart recovery
restart_signal="/app/temp/restart_requested.signal"
if [ -f "$restart_signal" ]; then
    log "✅ Restart recovery detected - processing restart signal"
    
    if [ "$DEBUG_STARTUP" = "true" ]; then
        log "Restart signal contents:"
        cat "$restart_signal" 2>/dev/null || log "Could not read restart signal"
    fi
    
    # log "GPU memory after restart:"
    # if command -v nvidia-smi >/dev/null; then
    #     nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | while read line; do
    #         used=$(echo $line | cut -d',' -f1)
    #         total=$(echo $line | cut -d',' -f2)
    #         used_gb=$((used / 1024))
    #         total_gb=$((total / 1024))
    #         log "GPU: ${used_gb}GB used / ${total_gb}GB total"
    #     done
    # else
    #     log "nvidia-smi not available"
    # fi
    
    # NOTE: Don't remove here - let Python app handle it
else
    log "Normal startup (no restart signal)"
fi

# Basic system checks
log "Checking system requirements..."

# Check Python
if ! python3 --version >/dev/null 2>&1; then
    log_error "Python3 not available"
    exit 1
fi

# Check main.py
if [ ! -f "main.py" ]; then
    log_error "main.py not found in $(pwd)"
    ls -la . || true
    exit 1
fi

# Test critical imports with timeout
log "Testing critical imports..."
timeout 60 python3 -c "
import sys
sys.path.insert(0, '/app')

# Test basic imports
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')

if torch.cuda.is_available():
    print(f'CUDA devices: {torch.cuda.device_count()}')
    try:
        props = torch.cuda.get_device_properties(0)
        print(f'GPU: {props.name}')
    except Exception as e:
        print(f'GPU props error: {e}')

# Test app imports
try:
    from app.config import Config
    print('✅ Config import OK')
except Exception as e:
    print(f'❌ Config import failed: {e}')
    sys.exit(1)

try:
    from app.models.llama_manager import LlamaManager
    print('✅ LlamaManager import OK')
except Exception as e:
    print(f'❌ LlamaManager import failed: {e}')

try:
    from app.services.llm_training_manager import LLMTrainingManager
    print('✅ Training Manager import OK')
except Exception as e:
    print(f'❌ Training Manager import failed: {e}')

print('✅ Critical imports completed')
" || {
    log_error "Critical imports test failed"
    if [ "$DEBUG_STARTUP" = "true" ]; then
        log_error "Python sys.path:"
        python3 -c "import sys; print('\n'.join(sys.path))" || true
    fi
    exit 1
}

# Set final environment
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM="false"
export TRANSFORMERS_CACHE="/app/.cache"

log "Environment ready - starting application"

if [ "$DEBUG_STARTUP" = "true" ]; then
    log "Final environment check:"
    log "PYTHONPATH: $PYTHONPATH"
    log "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
    log "Working directory: $(pwd)"
fi

# Start application
log "Executing: python3 main.py"
exec python3 main.py


