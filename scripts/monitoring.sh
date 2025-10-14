#!/bin/bash
# monitoring.sh - GPU memory monitoring for debugging training issues

LOG_FILE="/app/logs/gpu_monitoring.log"
ALERT_THRESHOLD=15  # Alert if GPU memory drops below 15GB

log_with_timestamp() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

get_gpu_memory() {
    nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1
}

get_gpu_used() {
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1
}

get_processes() {
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null
}

monitor_system() {
    log_with_timestamp "=== GPU Memory Monitoring Started ==="
    
    while true; do
        FREE_MEM=$(get_gpu_memory)
        USED_MEM=$(get_gpu_used)
        
        if [[ -n "$FREE_MEM" && -n "$USED_MEM" ]]; then
            FREE_GB=$((FREE_MEM / 1024))
            USED_GB=$((USED_MEM / 1024))
            TOTAL_GB=$((FREE_GB + USED_GB))
            
            log_with_timestamp "GPU Memory: ${USED_GB}GB used, ${FREE_GB}GB free (Total: ${TOTAL_GB}GB)"
            
            # Alert if memory is low
            if [[ $FREE_GB -lt $ALERT_THRESHOLD ]]; then
                log_with_timestamp "⚠️ ALERT: Low GPU memory detected (${FREE_GB}GB < ${ALERT_THRESHOLD}GB)"
                log_with_timestamp "Active GPU processes:"
                get_processes | while read line; do
                    log_with_timestamp "  $line"
                done
            fi
            
            # Container stats
            CONTAINER_MEM=$(docker stats llama-flask-backend --no-stream --format "table {{.MemUsage}}" 2>/dev/null | tail -1)
            log_with_timestamp "Container Memory: $CONTAINER_MEM"
            
        else
            log_with_timestamp "❌ Error: Could not get GPU memory info"
        fi
        
        sleep 30  # Monitor every 30 seconds
    done
}

# Cleanup function for training failures
cleanup_stuck_processes() {
    log_with_timestamp "🧹 Cleaning up stuck processes..."
    
    # Kill any stuck Python training processes
    pkill -f "python.*training" 2>/dev/null || true
    pkill -f "transformers" 2>/dev/null || true
    
    # Clear GPU memory
    nvidia-smi --gpu-reset 2>/dev/null || true
    
    # Restart container if needed
    docker restart llama-flask-backend 2>/dev/null || true
    
    log_with_timestamp "✅ Cleanup completed"
}

# Force memory reset
force_gpu_reset() {
    log_with_timestamp "🔥 FORCE GPU RESET REQUESTED"
    
    # Stop the backend container
    docker stop llama-flask-backend
    
    # Reset GPU
    nvidia-smi --gpu-reset
    
    # Clear system memory
    sync && echo 3 > /proc/sys/vm/drop_caches
    
    # Restart container
    docker start llama-flask-backend
    
    log_with_timestamp "🔥 Force reset completed"
}

case "$1" in
    "monitor")
        monitor_system
        ;;
    "cleanup")
        cleanup_stuck_processes
        ;;
    "reset")
        force_gpu_reset
        ;;
    *)
        echo "Usage: $0 {monitor|cleanup|reset}"
        echo "  monitor - Start continuous GPU monitoring"
        echo "  cleanup - Clean up stuck processes"
        echo "  reset   - Force GPU reset and container restart"
        exit 1
        ;;
esac