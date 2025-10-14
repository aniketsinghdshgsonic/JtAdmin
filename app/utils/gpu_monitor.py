# # utils/gpu_monitor.py
# """
# GPU Memory Monitoring Utility
# Simple GPU monitoring that integrates with existing logging
# """

# import logging
# import threading
# import time
# from typing import Optional, Dict, Any

# try:
#     import pynvml
#     PYNVML_AVAILABLE = True
# except ImportError:
#     PYNVML_AVAILABLE = False
#     pynvml = None

# logger = logging.getLogger(__name__)

# class GPUMonitor:
#     """Lightweight GPU monitoring utility"""
    
#     def __init__(self):
#         self.is_initialized = False
#         self.gpu_handle = None
#         self.gpu_name = "Unknown"
#         self.total_memory = 0
        
#         if PYNVML_AVAILABLE:
#             try:
#                 pynvml.nvmlInit()
#                 self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # First GPU
#                 self.gpu_name = pynvml.nvmlDeviceGetName(self.gpu_handle).decode('utf-8')
#                 mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
#                 self.total_memory = mem_info.total / 1024**3  # Convert to GB
#                 self.is_initialized = True
#                 logger.info(f"✅ GPU Monitor initialized: {self.gpu_name} ({self.total_memory:.1f}GB)")
#             except Exception as e:
#                 logger.warning(f"⚠️ GPU monitoring not available: {e}")
#         else:
#             logger.warning("⚠️ nvidia-ml-py not available, GPU monitoring disabled")
    
#     def get_gpu_usage(self) -> Dict[str, Any]:
#         """Get current GPU usage statistics"""
#         if not self.is_initialized:
#             return {"available": False}
        
#         try:
#             # Get memory info
#             mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
#             used_memory_gb = mem_info.used / 1024**3
#             memory_percent = (mem_info.used / mem_info.total) * 100
            
#             # Get utilization
#             util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
            
#             # Get temperature
#             try:
#                 temp = pynvml.nvmlDeviceGetTemperature(self.gpu_handle, pynvml.NVML_TEMPERATURE_GPU)
#             except:
#                 temp = 0
            
#             return {
#                 "available": True,
#                 "gpu_name": self.gpu_name,
#                 "memory_used_gb": round(used_memory_gb, 2),
#                 "memory_total_gb": round(self.total_memory, 2),
#                 "memory_percent": round(memory_percent, 1),
#                 "gpu_utilization": util.gpu,
#                 "memory_utilization": util.memory,
#                 "temperature": temp
#             }
#         except Exception as e:
#             logger.error(f"Error getting GPU stats: {e}")
#             return {"available": False, "error": str(e)}
    
#     def log_gpu_usage(self, context: str = ""):
#         """Log current GPU usage with context"""
#         stats = self.get_gpu_usage()
#         if stats.get("available"):
#             context_str = f" [{context}]" if context else ""
#             logger.info(
#                 f"🎮 GPU{context_str}: {stats['memory_used_gb']:.2f}GB/"
#                 f"{stats['memory_total_gb']:.1f}GB ({stats['memory_percent']:.1f}%) | "
#                 f"Util: {stats['gpu_utilization']}% | Temp: {stats['temperature']}°C"
#             )
#         else:
#             logger.debug(f"GPU monitoring not available{context_str}")
    
#     def log_memory_change(self, before_context: str, after_context: str):
#         """Log memory usage before and after an operation"""
#         before_stats = self.get_gpu_usage()
#         return before_stats  # Return for comparison later
    
#     def compare_and_log(self, before_stats: Dict, context: str):
#         """Compare before/after stats and log the difference"""
#         after_stats = self.get_gpu_usage()
        
#         if before_stats.get("available") and after_stats.get("available"):
#             memory_diff = after_stats["memory_used_gb"] - before_stats["memory_used_gb"]
#             diff_sign = "+" if memory_diff > 0 else ""
#             logger.info(
#                 f"🎮 GPU Memory Change [{context}]: {diff_sign}{memory_diff:.2f}GB | "
#                 f"Now: {after_stats['memory_used_gb']:.2f}GB/{after_stats['memory_total_gb']:.1f}GB "
#                 f"({after_stats['memory_percent']:.1f}%)"
#             )

# # Global instance
# gpu_monitor = GPUMonitor()








# app/utils/gpu_monitor.py
"""
GPU Memory Monitoring Utility
Simple GPU monitoring that integrates with existing logging
"""

import logging
import threading
import time
from typing import Optional, Dict, Any

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    pynvml = None

logger = logging.getLogger(__name__)

class GPUMonitor:
    """Lightweight GPU monitoring utility"""
    
    def __init__(self):
        self.is_initialized = False
        self.gpu_handle = None
        self.gpu_name = "Unknown"
        self.total_memory = 0
        
        if PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # First GPU
                self.gpu_name = pynvml.nvmlDeviceGetName(self.gpu_handle).decode('utf-8')
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                self.total_memory = mem_info.total / 1024**3  # Convert to GB
                self.is_initialized = True
                logger.info(f"✅ GPU Monitor initialized: {self.gpu_name} ({self.total_memory:.1f}GB)")
            except Exception as e:
                logger.warning(f"⚠️ GPU monitoring not available: {e}")
        else:
            logger.warning("⚠️ nvidia-ml-py not available, GPU monitoring disabled")
    
    def get_gpu_usage(self) -> Dict[str, Any]:
        """Get current GPU usage statistics"""
        if not self.is_initialized:
            return {"available": False}
        
        try:
            # Get memory info
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
            used_memory_gb = mem_info.used / 1024**3
            memory_percent = (mem_info.used / mem_info.total) * 100
            
            # Get utilization
            util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
            
            # Get temperature
            try:
                temp = pynvml.nvmlDeviceGetTemperature(self.gpu_handle, pynvml.NVML_TEMPERATURE_GPU)
            except:
                temp = 0
            
            return {
                "available": True,
                "gpu_name": self.gpu_name,
                "memory_used_gb": round(used_memory_gb, 2),
                "memory_total_gb": round(self.total_memory, 2),
                "memory_percent": round(memory_percent, 1),
                "gpu_utilization": util.gpu,
                "memory_utilization": util.memory,
                "temperature": temp
            }
        except Exception as e:
            logger.error(f"Error getting GPU stats: {e}")
            return {"available": False, "error": str(e)}
    
    def log_gpu_usage(self, context: str = ""):
        """Log current GPU usage with context"""
        stats = self.get_gpu_usage()
        context_str = f" [{context}]" if context else ""
        
        if stats.get("available"):
            logger.info(
                f"🎮 GPU{context_str}: {stats['memory_used_gb']:.2f}GB/"
                f"{stats['memory_total_gb']:.1f}GB ({stats['memory_percent']:.1f}%) | "
                f"Util: {stats['gpu_utilization']}% | Temp: {stats['temperature']}°C"
            )
        else:
            logger.debug(f"GPU monitoring not available{context_str}")
    
    def log_memory_change(self, before_context: str, after_context: str):
        """Log memory usage before and after an operation"""
        before_stats = self.get_gpu_usage()
        return before_stats  # Return for comparison later
    
    def compare_and_log(self, before_stats: Dict, context: str):
        """Compare before/after stats and log the difference"""
        after_stats = self.get_gpu_usage()
        
        if before_stats.get("available") and after_stats.get("available"):
            memory_diff = after_stats["memory_used_gb"] - before_stats["memory_used_gb"]
            diff_sign = "+" if memory_diff > 0 else ""
            logger.info(
                f"🎮 GPU Memory Change [{context}]: {diff_sign}{memory_diff:.2f}GB | "
                f"Now: {after_stats['memory_used_gb']:.2f}GB/{after_stats['memory_total_gb']:.1f}GB "
                f"({after_stats['memory_percent']:.1f}%)"
            )

# Global instance
gpu_monitor = GPUMonitor()