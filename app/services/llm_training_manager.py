# llm_training_manager.py - Fixed GPU Memory Management for Auto-Continual Learning
import os , sys
import json
import uuid
import time
import threading
import queue
import subprocess
import gc
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging
import torch
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    TrainerCallback,
    LlamaForCausalLM,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training,
    PeftModel,
)
from datasets import Dataset
import random
from sklearn.model_selection import train_test_split
from app.models.llama_manager import LlamaManager

logger = logging.getLogger(__name__)


class TrainingJob:
    def __init__(self, job_id: str, job_name: str = ""):
        self.job_id = job_id
        self.job_name = job_name
        self.status = "pending"  # pending, running, completed, failed
        self.progress = 0.0
        self.current_epoch = 0
        self.total_epochs = 0
        self.loss = None
        self.validation_loss = None
        self.best_loss = float("inf")
        self.message = "Job created"
        self.start_time = datetime.now()
        self.end_time = None
        self.model_path = None
        self.output_path = None
        self.training_metrics = {}
        self.validation_metrics = {}
        self.data_analysis = {}
        self.model_version = 1
        self.base_job_id = None
        self.training_type = "fresh"
        self.auto_continued = False
        # NEW: Track if model is loaded in memory
        self._model = None
        self._tokenizer = None


class EnhancedProgressCallback(TrainerCallback):
    def __init__(self, job, job_manager):
        self.job = job
        self.job_manager = job_manager
        self.step_count = 0
        self.best_eval_loss = float("inf")

    def on_init_end(self, args, state, control, **kwargs):
        pass

    def on_epoch_end(self, args, state, control, **kwargs):
        self.job.current_epoch = int(state.epoch)
        progress = 70.0 + (state.epoch / args.num_train_epochs) * 25.0
        self.job.progress = min(progress, 95.0)
        self.job.message = f"Epoch {int(state.epoch)}/{args.num_train_epochs}"

        self.job.training_metrics[f"epoch_{int(state.epoch)}"] = {
            "epoch": int(state.epoch),
            "learning_rate": state.log_history[-1].get("learning_rate", 0)
            if state.log_history
            else 0,
            "train_loss": state.log_history[-1].get("train_loss", 0)
            if state.log_history
            else 0,
        }

        # Enhanced GPU cleanup during training
        self._enhanced_gpu_cleanup()

        # Save progress to disk
        self.job_manager._save_jobs_to_disk()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            if "loss" in logs:
                self.job.loss = logs["loss"]
            if "eval_loss" in logs:
                self.job.validation_loss = logs["eval_loss"]
                if logs["eval_loss"] < self.job.best_loss:
                    self.job.best_loss = logs["eval_loss"]

            self.step_count += 1
            self.job.validation_metrics[f"step_{self.step_count}"] = logs.copy()

    def _enhanced_gpu_cleanup(self):
        """Enhanced GPU cleanup with proper error handling"""
        try:
            if torch.cuda.is_available():
                # Force cleanup of any unused tensors
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

                # Additional cleanup
                if hasattr(torch.cuda, "reset_max_memory_allocated"):
                    torch.cuda.reset_max_memory_allocated()

                # Force garbage collection
                gc.collect()

        except Exception as e:
            logger.warning(f"Enhanced GPU cleanup warning: {e}")


class LLMTrainingManager:
    """Fixed Auto-Continual Learning Training Manager with Proper Memory Management"""

    def __init__(self, ai_model_dir: str = "/app/ai-model" , llama_manager: Optional['LlamaManager'] = None):
        self.ai_model_dir = ai_model_dir
        self.jobs: Dict[str, TrainingJob] = {}
        self.llama_manager = llama_manager
        self.trained_models: Dict[str, tuple] = {}
        self.active_training = False
        self.cuda_available = False
        self.cuda_error = None

        # Job persistence and cleanup settings
        self.jobs_file = f"{ai_model_dir}/training_jobs.json"
        self.max_models_to_keep = 3  # Keep only 3 most recent successful models
        self.max_models_in_memory = 1  # NEW: Only keep 1 model in GPU memory at a time
        self.restart_after_training = (
            os.getenv("RESTART_AFTER_TRAINING", "true").lower() == "true"
        )
        self.restart_threshold = int(
            os.getenv("RESTART_TRAINING_THRESHOLD", "1")
        )  # Restart after every training

        # Initialize CUDA environment safely
        self._initialize_cuda_environment()

        # Ensure directories exist
        os.makedirs(f"{ai_model_dir}/trained_models", exist_ok=True)
        os.makedirs(f"{ai_model_dir}/training_data", exist_ok=True)
        os.makedirs(f"{ai_model_dir}/model_evaluations", exist_ok=True)

        # Load persisted jobs on startup
        self._load_persisted_jobs()

        logger.info("Fixed Auto-Continual Learning Training Manager initialized")
        logger.info(f"CUDA Available: {self.cuda_available}")
        logger.info(f"Loaded {len(self.jobs)} persisted training jobs")
        logger.info(f"Max models in memory: {self.max_models_in_memory}")

        # Clean up old models on startup
        self._cleanup_old_models()

    def _initialize_cuda_environment(self):
        """Initialize CUDA environment with proper error handling"""
        try:
            # Set environment variables to prevent NVML issues
            os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

            # Check CUDA availability
            self.cuda_available = torch.cuda.is_available()

            if self.cuda_available:
                # Initialize CUDA context safely
                torch.cuda.init()

                # Get device properties
                device_count = torch.cuda.device_count()
                if device_count > 0:
                    device_props = torch.cuda.get_device_properties(0)
                    logger.info(f"CUDA Device: {device_props.name}")
                    logger.info(
                        f"CUDA Memory: {device_props.total_memory / 1024**3:.2f} GB"
                    )

                    # Test basic CUDA operations
                    test_tensor = torch.ones(1).cuda()
                    test_result = test_tensor + 1
                    logger.info("CUDA initialization test passed")

                    # Clean up test tensor
                    del test_tensor, test_result
                    torch.cuda.empty_cache()
                else:
                    self.cuda_available = False
                    self.cuda_error = "No CUDA devices found"
            else:
                self.cuda_error = "CUDA not available in PyTorch"

        except Exception as e:
            self.cuda_available = False
            self.cuda_error = f"CUDA initialization failed: {str(e)}"
            logger.error(f"CUDA initialization error: {e}")

    def _save_jobs_to_disk(self):
        """Persist training jobs to disk for container restart recovery"""
        try:
            jobs_data = {}
            for job_id, job in self.jobs.items():
                jobs_data[job_id] = {
                    "job_id": job.job_id,
                    "job_name": job.job_name,
                    "status": job.status,
                    "progress": job.progress,
                    "current_epoch": job.current_epoch,
                    "total_epochs": job.total_epochs,
                    "loss": job.loss,
                    "validation_loss": job.validation_loss,
                    "best_loss": job.best_loss,
                    "message": job.message,
                    "start_time": job.start_time.isoformat(),
                    "end_time": job.end_time.isoformat() if job.end_time else None,
                    "model_path": job.model_path,
                    "output_path": job.output_path,
                    "training_metrics": job.training_metrics,
                    "validation_metrics": job.validation_metrics,
                    "data_analysis": job.data_analysis,
                    "model_version": job.model_version,
                    "base_job_id": job.base_job_id,
                    "training_type": job.training_type,
                    "auto_continued": job.auto_continued,
                }

            with open(self.jobs_file, "w") as f:
                json.dump(jobs_data, f, indent=2)

            logger.debug(f"Saved {len(jobs_data)} jobs to disk")

        except Exception as e:
            logger.warning(f"Failed to save jobs to disk: {e}")

    def _load_persisted_jobs(self):
        """Load persisted training jobs from disk"""
        try:
            if not os.path.exists(self.jobs_file):
                logger.info("No persisted jobs file found - starting fresh")
                return

            with open(self.jobs_file, "r") as f:
                jobs_data = json.load(f)

            for job_id, job_data in jobs_data.items():
                try:
                    job = TrainingJob(job_data["job_id"], job_data["job_name"])
                    job.status = job_data["status"]
                    job.progress = job_data["progress"]
                    job.current_epoch = job_data["current_epoch"]
                    job.total_epochs = job_data["total_epochs"]
                    job.loss = job_data["loss"]
                    job.validation_loss = job_data["validation_loss"]
                    job.best_loss = job_data["best_loss"]
                    job.message = job_data["message"]
                    job.start_time = datetime.fromisoformat(job_data["start_time"])
                    job.end_time = (
                        datetime.fromisoformat(job_data["end_time"])
                        if job_data["end_time"]
                        else None
                    )
                    job.model_path = job_data["model_path"]
                    job.output_path = job_data["output_path"]
                    job.training_metrics = job_data["training_metrics"]
                    job.validation_metrics = job_data["validation_metrics"]
                    job.data_analysis = job_data["data_analysis"]
                    job.model_version = job_data["model_version"]
                    job.base_job_id = job_data["base_job_id"]
                    job.training_type = job_data["training_type"]
                    job.auto_continued = job_data["auto_continued"]

                    self.jobs[job_id] = job

                except Exception as job_error:
                    logger.warning(f"Failed to load job {job_id}: {job_error}")

            logger.info(f"Successfully loaded {len(self.jobs)} persisted jobs")

            # Log the latest successful job for debugging
            latest_job_id = self.get_latest_successful_job()
            if latest_job_id:
                latest_job = self.jobs[latest_job_id]
                logger.info(
                    f"Latest successful job: {latest_job_id} ({latest_job.job_name}) - v{latest_job.model_version}"
                )
            else:
                logger.info("No successful jobs found in persisted data")

        except Exception as e:
            logger.error(f"Failed to load persisted jobs: {e}")
            self.jobs = {}

    def _cleanup_old_models(self):
        """Clean up old model directories, keeping only the most recent successful ones"""
        try:
            trained_models_dir = f"{self.ai_model_dir}/trained_models"

            if not os.path.exists(trained_models_dir):
                logger.info("No trained models directory found")
                return

            # Get all completed jobs sorted by completion time (newest first)
            completed_jobs = [
                job
                for job in self.jobs.values()
                if job.status == "completed" and job.end_time
            ]

            if len(completed_jobs) <= self.max_models_to_keep:
                logger.info(
                    f"Have {len(completed_jobs)} completed jobs, no cleanup needed"
                )
                return

            # Sort by completion time (newest first)
            completed_jobs.sort(key=lambda x: x.end_time, reverse=True)

            # Keep the most recent models, remove the rest
            jobs_to_keep = completed_jobs[: self.max_models_to_keep]
            jobs_to_remove = completed_jobs[self.max_models_to_keep :]

            logger.info(
                f"Cleaning up {len(jobs_to_remove)} old model(s), keeping {len(jobs_to_keep)} most recent"
            )

            total_space_freed = 0

            for job in jobs_to_remove:
                model_dir = f"{trained_models_dir}/{job.job_id}"

                if os.path.exists(model_dir):
                    try:
                        # Calculate size before deletion
                        dir_size = sum(
                            os.path.getsize(os.path.join(dirpath, filename))
                            for dirpath, dirnames, filenames in os.walk(model_dir)
                            for filename in filenames
                        )

                        # Remove the directory
                        shutil.rmtree(model_dir)
                        total_space_freed += dir_size

                        logger.info(
                            f"Removed old model: {job.job_id} ({job.job_name}) - freed {dir_size / 1024**2:.1f} MB"
                        )

                    except Exception as remove_error:
                        logger.warning(
                            f"Failed to remove model directory {model_dir}: {remove_error}"
                        )

            if total_space_freed > 0:
                logger.info(
                    f"Model cleanup completed: freed {total_space_freed / 1024**2:.1f} MB total"
                )

            # Log what we're keeping
            logger.info("Keeping these models:")
            for i, job in enumerate(jobs_to_keep):
                logger.info(
                    f"  {i+1}. {job.job_id} ({job.job_name}) - v{job.model_version} - {job.end_time.strftime('%Y-%m-%d %H:%M')}"
                )

        except Exception as e:
            logger.error(f"Model cleanup failed: {e}")



    def _should_restart_container(self) -> bool:
        """FIXED: Check if container should restart after this training"""
        if not self.restart_after_training:
            logger.debug("Container restart disabled by configuration")
            return False

        try:
            completed_jobs = [j for j in self.jobs.values() if j.status == "completed"]
            training_count = len(completed_jobs)

            # Check available GPU memory
            available_memory = self._get_available_gpu_memory()
            
            # More intelligent restart criteria
            memory_based_restart = available_memory < 15.0  # Less than 15GB available
            count_based_restart = training_count > 0 and training_count % self.restart_threshold == 0
            
            should_restart = memory_based_restart or count_based_restart
            
            if should_restart:
                if memory_based_restart:
                    logger.info(
                        f"Container restart triggered by low GPU memory: {available_memory:.1f}GB < 15GB"
                    )
                if count_based_restart:
                    logger.info(
                        f"Container restart triggered by training count: {training_count} (threshold: {self.restart_threshold})"
                    )
            else:
                logger.debug(
                    f"No restart needed: {training_count} trainings, {available_memory:.1f}GB GPU memory"
                )

            return should_restart

        except Exception as e:
            logger.error(f"Error checking restart conditions: {e}")
            # Default to restart on error for safety
            return True



    def _schedule_container_restart(self, job_id: str):
        """FIXED: Properly restart the entire container"""
        try:
            logger.info("🔄 Scheduling container restart for complete GPU memory recovery...")

            # 1. Save all job data multiple times for safety
            for i in range(3):
                try:
                    self._save_jobs_to_disk()
                    logger.debug(f"Jobs saved to disk (attempt {i+1}/3)")
                    time.sleep(0.5)
                except Exception as save_error:
                    logger.warning(f"Job save attempt {i+1} failed: {save_error}")

            # 2. Create restart signal file
            restart_signal_file = "/app/temp/restart_requested.signal"
            os.makedirs("/app/temp", exist_ok=True)

            restart_info = {
                "restart_requested": True,
                "reason": "post_training_memory_recovery", 
                "job_id": job_id,
                "timestamp": datetime.now().isoformat(),
                "completed_trainings": len([j for j in self.jobs.values() if j.status == "completed"]),
                "restart_method": "container_exit"
            }

            with open(restart_signal_file, "w") as f:
                json.dump(restart_info, f, indent=2)
                
            logger.info("✅ Restart signal created")
            logger.info("🔄 Container will exit in 10 seconds for Docker restart...")

            # 3. Schedule container exit (not just Python exit)
            def delayed_container_exit():
                try:
                    time.sleep(10)  # Wait for API response
                    logger.info("🔄 Initiating container exit for restart...")
                    
                    # Final cleanup
                    try:
                        self._force_complete_gpu_reset()
                        self._save_jobs_to_disk()
                    except Exception as cleanup_error:
                        logger.warning(f"Pre-exit cleanup error: {cleanup_error}")

                    # FIXED: Exit the container properly
                    # Method 1: Send SIGTERM to container's main process (PID 1)
                    try:
                        import subprocess
                        # Kill the main container process (bash script)
                        subprocess.run(['kill', '-TERM', '1'], check=False)
                        time.sleep(2)
                        # Force kill if still running
                        subprocess.run(['kill', '-KILL', '1'], check=False)
                    except Exception as kill_error:
                        logger.warning(f"Kill PID 1 failed: {kill_error}")
                    
                    # Method 2: Multiple exit strategies as fallbacks
                    exit_methods = [
                        ("os._exit(0)", lambda: os._exit(0)),    # Normal exit
                        ("sys.exit(0)", lambda: sys.exit(0)),   # Python exit
                        ("os._exit(1)", lambda: os._exit(1)),   # Error exit
                    ]
                    
                    for method_name, method_func in exit_methods:
                        try:
                            logger.info(f"Container exit method: {method_name}")
                            method_func()
                            break
                        except Exception as exit_error:
                            logger.error(f"Exit method {method_name} failed: {exit_error}")
                            time.sleep(1)

                except Exception as e:
                    logger.error(f"Container exit process failed: {e}")
                    # Emergency exit
                    os._exit(1)

            # Start exit thread
            restart_thread = threading.Thread(target=delayed_container_exit, name="ContainerExitThread")
            restart_thread.daemon = True
            restart_thread.start()
            
            logger.info("✅ Container exit scheduled")

        except Exception as e:
            logger.error(f"Failed to schedule container restart: {e}", exc_info=True)




    def _unload_models_from_memory(self, exclude_job_id: str = None):
        """Properly unload models with aggressive GPU cleanup"""
        logger.info("Performing aggressive GPU memory cleanup...")

        models_to_remove = []
        for job_id in list(self.trained_models.keys()):
            if job_id != exclude_job_id:
                models_to_remove.append(job_id)

        if not models_to_remove:
            logger.info("No models to unload")
            return

        for job_id in models_to_remove:
            try:
                logger.info(f"Aggressively unloading model {job_id}...")

                if job_id in self.trained_models:
                    model, tokenizer = self.trained_models[job_id]

                    # Move model to CPU first
                    if hasattr(model, "cpu"):
                        model.cpu()

                    # Clear CUDA cache for this model
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()

                    # Delete references
                    del model, tokenizer
                    del self.trained_models[job_id]

                    # Force garbage collection
                    import gc

                    gc.collect()

                    # Additional CUDA cleanup
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                        torch.cuda.ipc_collect()

                    logger.info(f"✅ Aggressively cleaned model {job_id}")

            except Exception as e:
                logger.warning(
                    f"Error during aggressive cleanup of model {job_id}: {e}"
                )

        # Final system-wide cleanup
        try:
            import gc

            for _ in range(3):  # Multiple cleanup passes
                gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                torch.cuda.reset_max_memory_allocated()
                torch.cuda.reset_peak_memory_stats()

                # Nuclear option: reset CUDA context
                torch.cuda.ipc_collect()

        except Exception as e:
            logger.warning(f"Final cleanup warning: {e}")

        logger.info(
            f"✅ Aggressive cleanup completed. Models in memory: {len(self.trained_models)}"
        )

    def _force_complete_gpu_reset(self):
        """Nuclear GPU memory reset"""
        try:
            logger.info("🔥 NUCLEAR GPU RESET - Clearing everything...")

            # Clear all models from memory first
            self.trained_models.clear()

            # Multiple garbage collection passes
            import gc

            for i in range(5):
                gc.collect()
                time.sleep(0.5)

            if torch.cuda.is_available():
                # Multiple CUDA cleanup passes
                for i in range(3):
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    time.sleep(1)

                # Reset all CUDA memory tracking
                torch.cuda.reset_max_memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.ipc_collect()

                # Force CUDA context recreation
                try:
                    # This forces complete CUDA context reset
                    import subprocess

                    subprocess.run(["nvidia-smi", "--gpu-reset"], capture_output=True)
                except:
                    pass  # Fallback if nvidia-smi not available

            logger.info("🔥 Nuclear GPU reset completed")

        except Exception as e:
            logger.error(f"Nuclear GPU reset failed: {e}")

    def _get_available_gpu_memory(self):
        """Get available GPU memory in GB"""
        if not torch.cuda.is_available():
            return 0

        try:
            device = torch.cuda.current_device()
            total_memory = torch.cuda.get_device_properties(device).total_memory
            allocated_memory = torch.cuda.memory_allocated(device)
            available_memory = total_memory - allocated_memory

            return available_memory / (1024**3)  # Convert to GB
        except Exception as e:
            logger.warning(f"Could not get GPU memory info: {e}")
            return 0

    def _adjust_memory_config_for_continual(self, base_config: dict) -> dict:
        """Adjust memory configuration for continual learning based on available memory"""
        available_memory = self._get_available_gpu_memory()
        logger.info(f"Available GPU memory: {available_memory:.1f} GB")

        adjusted_config = base_config.copy()

        if available_memory < 12:
            # Reduce GPU allocation if limited memory
            adjusted_config["max_gpu"] = f"{max(8, int(available_memory * 0.8))}GB"
            adjusted_config["max_cpu"] = (
                f"{min(50, 40 + int((12 - available_memory) * 2))}GB"
            )
            logger.info(
                f"Adjusted memory config for continual learning: GPU={adjusted_config['max_gpu']}, CPU={adjusted_config['max_cpu']}"
            )

        return adjusted_config

    def _load_llama_with_fallback(self, model_path: str, **load_kwargs):
        """Load a LLaMA family model with AutoModel fallback handling."""

        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        try:
            return AutoModelForCausalLM.from_pretrained(
                model_path, config=config, trust_remote_code=True, **load_kwargs
            )
        except (ValueError, OSError) as load_error:
            error_text = str(load_error)
            if "Unrecognized model" not in error_text and "model_type" not in error_text:
                raise

            logger.warning(
                "AutoModel could not identify the model (%s). Falling back to LlamaForCausalLM.",
                error_text,
            )
            return LlamaForCausalLM.from_pretrained(
                model_path, config=config, trust_remote_code=True, **load_kwargs
            )

    def get_latest_successful_job(self) -> Optional[str]:
        """Find the most recent successfully completed training job with robust fallback"""

        completed_jobs = [
            job for job in self.jobs.values() if job.status == "completed"
        ]

        if not completed_jobs:
            logger.info("No completed jobs found - will start fresh training")
            return None

        # Sort by end_time to get the most recent
        completed_jobs_sorted = sorted(
            completed_jobs, key=lambda job: job.end_time, reverse=True
        )

        # Try each job in order until we find one with valid files
        required_files = ["adapter_config.json", "adapter_model.safetensors"]

        for job in completed_jobs_sorted:
            adapter_path = f"{self.ai_model_dir}/trained_models/{job.job_id}"

            # Check if all required files exist
            valid_files = []
            missing_files = []

            for file in required_files:
                file_path = os.path.join(adapter_path, file)
                if os.path.exists(file_path):
                    valid_files.append(file)
                else:
                    missing_files.append(file)

            if not missing_files:
                # All files present - this is our candidate
                logger.info(
                    f"Auto-detected latest successful job: {job.job_id} ({job.job_name}) - v{job.model_version}"
                )
                return job.job_id
            else:
                logger.warning(
                    f"Job {job.job_id} missing files: {missing_files}, trying next most recent..."
                )

        # If we get here, no valid completed jobs were found
        logger.warning(
            "No valid completed jobs found with required adapter files - will start fresh"
        )
        return None

    def get_all_valid_jobs_for_continual_learning(self) -> List[str]:
        """Get all valid jobs that can be used for continual learning, sorted by preference"""

        completed_jobs = [
            job for job in self.jobs.values() if job.status == "completed"
        ]

        if not completed_jobs:
            return []

        # Sort by end_time (newest first)
        completed_jobs_sorted = sorted(
            completed_jobs, key=lambda job: job.end_time, reverse=True
        )

        valid_jobs = []
        required_files = ["adapter_config.json", "adapter_model.safetensors"]

        for job in completed_jobs_sorted:
            adapter_path = f"{self.ai_model_dir}/trained_models/{job.job_id}"

            # Check if all required files exist
            if all(
                os.path.exists(os.path.join(adapter_path, file))
                for file in required_files
            ):
                valid_jobs.append(job.job_id)

        logger.info(f"Found {len(valid_jobs)} valid jobs for continual learning")
        return valid_jobs

    def create_training_job(
        self, job_name: str = "", base_job_id: str = None, auto_continued: bool = False
    ) -> str:
        """Create training job with auto-continual learning support"""
        job_id = str(uuid.uuid4())
        job = TrainingJob(job_id, job_name)

        if base_job_id:
            job.base_job_id = base_job_id
            job.training_type = "continual"
            job.auto_continued = auto_continued

            # Inherit version from base job and increment
            base_job = self.jobs.get(base_job_id)
            if base_job:
                job.model_version = base_job.model_version + 1
        else:
            job.training_type = "fresh"

        self.jobs[job_id] = job

        # Save jobs to disk immediately
        self._save_jobs_to_disk()

        auto_msg = " (auto-detected)" if auto_continued else ""
        logger.info(
            f"Created training job: {job_id} ({job_name}) - Type: {job.training_type}{auto_msg}"
        )
        if base_job_id:
            logger.info(f"  Continuing from base job: {base_job_id}")

        return job_id

    def get_job_status(self, job_id: str) -> Optional[Dict]:
        """Get enhanced job status with auto-continual info"""
        job = self.jobs.get(job_id)
        if not job:
            return None

        return {
            "job_id": job.job_id,
            "job_name": job.job_name,
            "status": job.status,
            "progress": job.progress,
            "current_epoch": job.current_epoch,
            "total_epochs": job.total_epochs,
            "loss": job.loss,
            "validation_loss": job.validation_loss,
            "best_loss": job.best_loss,
            "message": job.message,
            "start_time": job.start_time.isoformat(),
            "end_time": job.end_time.isoformat() if job.end_time else None,
            "model_path": job.model_path,
            "output_path": job.output_path,
            "training_metrics": job.training_metrics,
            "validation_metrics": job.validation_metrics,
            "data_analysis": job.data_analysis,
            "model_version": job.model_version,
            "training_type": job.training_type,
            "base_job_id": job.base_job_id,
            "auto_continued": job.auto_continued,
        }

    def list_jobs(self) -> List[Dict]:
        """List all jobs with auto-continual info"""
        return [self.get_job_status(job_id) for job_id in self.jobs.keys()]

    def validate_base_job(self, base_job_id: str) -> tuple[bool, str]:
        """Validate base job for continual learning"""
        if not base_job_id:
            return True, "No base job specified"

        if base_job_id not in self.jobs:
            return False, f"Base job {base_job_id} not found"

        base_job = self.jobs[base_job_id]

        if base_job.status != "completed":
            return (
                False,
                f"Base job {base_job_id} status is {base_job.status}, must be 'completed'",
            )

        base_model_path = f"{self.ai_model_dir}/trained_models/{base_job_id}"
        if not os.path.exists(base_model_path):
            return False, f"Base job model files not found at {base_model_path}"

        adapter_files = ["adapter_config.json", "adapter_model.safetensors"]
        missing_files = []
        for file in adapter_files:
            if not os.path.exists(os.path.join(base_model_path, file)):
                missing_files.append(file)

        if missing_files:
            return False, f"Base job missing adapter files: {missing_files}"

        return True, f"Base job {base_job_id} validated successfully"

    def analyze_training_data(self, examples: List[Dict]) -> Dict:
        """Training data analysis"""
        analysis = {
            "total_examples": len(examples),
            "mapping_types": {},
            "data_quality": {},
            "recommendations": [],
            "validation_results": {"valid": True, "errors": [], "warnings": []},
        }

        mapping_types = {}
        input_lengths = []
        output_lengths = []

        for i, example in enumerate(examples):
            try:
                if not all(key in example for key in ["input_data", "output_data"]):
                    analysis["validation_results"]["errors"].append(
                        f"Example {i}: Missing required fields"
                    )
                    analysis["validation_results"]["valid"] = False
                    continue

                mapping_type = example.get("mapping_type", "general")
                mapping_types[mapping_type] = mapping_types.get(mapping_type, 0) + 1

                input_str = json.dumps(example["input_data"])
                output_str = json.dumps(example["output_data"])

                input_lengths.append(len(input_str))
                output_lengths.append(len(output_str))

                if len(input_str) > 8000:
                    analysis["validation_results"]["warnings"].append(
                        f"Example {i}: Very long input ({len(input_str)} chars)"
                    )

                if not example["input_data"] or not example["output_data"]:
                    analysis["validation_results"]["errors"].append(
                        f"Example {i}: Empty input or output data"
                    )
                    analysis["validation_results"]["valid"] = False

            except Exception as e:
                analysis["validation_results"]["errors"].append(
                    f"Example {i}: Invalid format - {str(e)}"
                )
                analysis["validation_results"]["valid"] = False

        analysis["mapping_types"] = mapping_types

        if input_lengths:
            analysis["data_quality"] = {
                "avg_input_length": int(np.mean(input_lengths)),
                "avg_output_length": int(np.mean(output_lengths)),
                "max_input_length": max(input_lengths),
                "max_output_length": max(output_lengths),
                "median_input_length": int(np.median(input_lengths)),
                "median_output_length": int(np.median(output_lengths)),
            }

        if len(examples) < 50:
            analysis["recommendations"].append(
                {
                    "type": "warning",
                    "message": "Consider gathering more training examples (50-500 recommended for good accuracy)",
                }
            )

        return analysis

    def optimize_training_config(
        self, examples: List[Dict], base_config: Dict, is_continual: bool = False
    ) -> Dict:
        """Optimize config with continual learning adjustments"""
        num_examples = len(examples)
        mapping_types = len(set(ex.get("mapping_type", "general") for ex in examples))
        optimized_config = base_config.copy()

        # Establish conservative defaults before applying heuristics
        optimized_config.setdefault("max_length", 1024)
        optimized_config.setdefault("per_device_batch_size", 1)
        optimized_config.setdefault("gradient_accumulation_steps", 1)
        optimized_config.setdefault("lora_r", max(8, base_config.get("lora_r", 8)))

        # Base optimization
        if num_examples < 100:
            optimized_config.update(
                {
                    "epochs": max(5, optimized_config.get("epochs", 3)),
                    "learning_rate": min(
                        5e-4, optimized_config.get("learning_rate", 2e-4) * 1.5
                    ),
                    "lora_r": min(12, max(8, optimized_config.get("lora_r", 8))),
                    "batch_size": 1,
                    "warmup_ratio": 0.1,
                }
            )
        elif num_examples < 500:
            optimized_config.update(
                {
                    "epochs": optimized_config.get("epochs", 3),
                    "learning_rate": optimized_config.get("learning_rate", 2e-4),
                    "lora_r": min(12, max(8, optimized_config.get("lora_r", 8))),
                    "batch_size": min(2, optimized_config.get("batch_size", 1)),
                    "warmup_ratio": 0.05,
                }
            )
        else:
            optimized_config.update(
                {
                    "epochs": min(2, optimized_config.get("epochs", 3)),
                    "learning_rate": max(
                        1e-4, optimized_config.get("learning_rate", 2e-4) * 0.7
                    ),
                    "lora_r": min(10, max(8, optimized_config.get("lora_r", 8))),
                    "batch_size": min(4, optimized_config.get("batch_size", 1)),
                    "warmup_ratio": 0.03,
                }
            )

        if mapping_types > 3:
            optimized_config["epochs"] = max(optimized_config["epochs"], 4)
            optimized_config["lora_r"] = min(12, max(optimized_config["lora_r"], 10))

        # Auto-continual learning adjustments
        if is_continual:
            optimized_config["learning_rate"] = optimized_config["learning_rate"] * 0.6
            optimized_config["epochs"] = max(1, optimized_config["epochs"] - 1)
            optimized_config["warmup_ratio"] = min(
                0.15, optimized_config["warmup_ratio"] * 1.5
            )
            logger.info("Applied auto-continual learning optimizations")

        # Final safety clamps
        optimized_config["lora_r"] = max(4, min(int(optimized_config.get("lora_r", 8)), 12))
        optimized_config["batch_size"] = max(
            1, int(optimized_config.get("batch_size", 1))
        )
        optimized_config["max_length"] = max(
            256, min(int(optimized_config.get("max_length", 1024)), 1536)
        )
        optimized_config["per_device_batch_size"] = max(
            1, int(optimized_config.get("per_device_batch_size", 1))
        )
        optimized_config["gradient_accumulation_steps"] = max(
            1,
            int(
                optimized_config.get(
                    "gradient_accumulation_steps",
                    optimized_config.get("batch_size", 1),
                )
            ),
        )

        return optimized_config

    def start_training(
        self,
        model_name: str,
        training_examples: List[Dict],
        config: Dict,
        job_name: str = "",
        force_fresh: bool = False,
    ) -> str:
        """Enhanced start_training with aggressive memory management and CUDA corruption recovery"""

        if self.active_training:
            raise Exception(
                "Training already in progress. Only one training job at a time."
            )

        # CRITICAL: Detect CUDA corruption and force fresh training
        cuda_corruption_detected = False
        try:
            if self.llama_manager and self.llama_manager.is_model_loaded():
                logger.info("🧹 Unloading GGUF inference model to free GPU memory for training...")
                self.llama_manager.unload_model()
                time.sleep(5)  # Wait for cleanup to complete
                
                # Additional aggressive cleanup
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    time.sleep(2)
                    
                # Verify memory freed
                available_memory = self._get_available_gpu_memory()
                logger.info(f"✅ GPU memory available for training: {available_memory:.1f} GB")
        except Exception as e:
            logger.warning(f"Could not unload GGUF model: {e}")
    
        try:
            if torch.cuda.is_available():
                # Test basic CUDA operations
                test_tensor = torch.ones(1).cuda()
                test_result = test_tensor + 1
                del test_tensor, test_result
                torch.cuda.empty_cache()
        except Exception as cuda_test_error:
            logger.error(f"🚨 CUDA corruption detected: {cuda_test_error}")
            cuda_corruption_detected = True

        # Force fresh training if CUDA issues detected
        if cuda_corruption_detected:
            logger.warning(
                "🔧 CUDA corruption detected - forcing fresh training to recover"
            )
            force_fresh = True
            self._force_complete_gpu_reset()

        # CRITICAL: Aggressive cleanup before each training
        logger.info("🧹 Pre-training aggressive cleanup...")

        # Nuclear cleanup every 2nd training to prevent accumulation
        completed_jobs = [j for j in self.jobs.values() if j.status == "completed"]
        training_count = len(completed_jobs)

        if training_count > 0 and training_count % 2 == 0:
            logger.info("🔥 Performing nuclear cleanup (every 2nd training)...")
            self._force_complete_gpu_reset()
            time.sleep(10)  # Wait longer after nuclear reset
        else:
            self._unload_models_from_memory()
            time.sleep(5)

        # Check available memory and force fresh if too low
        available_memory = self._get_available_gpu_memory()
        logger.info(f"Available GPU memory after cleanup: {available_memory:.1f} GB")

        # Force fresh training if memory too low (increased threshold)
        if available_memory < 18.0:
            logger.warning(
                f"⚠️ Low GPU memory ({available_memory:.1f}GB) - forcing fresh training"
            )
            force_fresh = True
            self._force_complete_gpu_reset()
            time.sleep(10)
            available_memory = self._get_available_gpu_memory()
            logger.info(f"Memory after nuclear reset: {available_memory:.1f} GB")

        # ENHANCED: Auto-continual logic with corruption protection
        base_job_id = None
        auto_continued = False

        if not force_fresh:
            valid_jobs = self.get_all_valid_jobs_for_continual_learning()

            if valid_jobs and available_memory > 16.0:  # Increased threshold
                base_job_id = valid_jobs[0]
                auto_continued = True
                logger.info(
                    f"AUTO-CONTINUAL: Using {base_job_id} as base (GPU memory: {available_memory:.1f}GB)"
                )
            elif valid_jobs:
                logger.warning(
                    f"⚠️ Insufficient GPU memory ({available_memory:.1f}GB) for continual learning - using fresh training"
                )
                force_fresh = True
            else:
                logger.info("No valid previous models found - starting fresh training")
        else:
            logger.info("Fresh training requested or forced due to CUDA issues")

        # Create job with enhanced error recovery
        job_id = self.create_training_job(job_name, base_job_id, auto_continued)
        job = self.jobs[job_id]

        # Add CUDA corruption recovery flag
        job._cuda_recovery_mode = cuda_corruption_detected or force_fresh

        # Analyze training data
        job.data_analysis = self.analyze_training_data(training_examples)

        if not job.data_analysis["validation_results"]["valid"]:
            job.status = "failed"
            job.message = f"Data validation failed: {job.data_analysis['validation_results']['errors']}"
            job.end_time = datetime.now()
            self._save_jobs_to_disk()
            return job_id

        # Model path validation
        hf_model_path = f"{self.ai_model_dir}/CodeLlama-13b-hf"
        if not os.path.exists(hf_model_path):
            job.status = "failed"
            job.message = f"HuggingFace model not found at {hf_model_path}"
            job.end_time = datetime.now()
            self._save_jobs_to_disk()
            return job_id

        job.model_path = hf_model_path
        job.output_path = f"{self.ai_model_dir}/trained_models/{job_id}"

        # Optimize configuration with corruption recovery
        is_continual = base_job_id is not None and not cuda_corruption_detected
        optimized_config = self.optimize_training_config(
            training_examples, config, is_continual
        )

        # Force conservative settings if CUDA issues detected or low memory
        if cuda_corruption_detected or available_memory < 20.0:
            optimized_config.update(
                {
                    "batch_size": 1,
                    "epochs": min(2, optimized_config.get("epochs", 3)),
                    "learning_rate": optimized_config.get("learning_rate", 2e-4) * 0.7,
                    "lora_r": min(16, optimized_config.get("lora_r", 16)),
                }
            )
            if cuda_corruption_detected:
                logger.info("Applied CUDA recovery optimizations")
            else:
                logger.info("Applied low memory optimizations")

        # Start training thread
        training_thread = threading.Thread(
            target=self._run_enhanced_training,
            args=(
                job_id,
                hf_model_path,
                training_examples,
                optimized_config,
                base_job_id,
            ),
        )
        training_thread.daemon = True
        training_thread.start()

        return job_id

    def _force_cuda_reset(self):
        """Force complete CUDA context reset to prevent corruption"""
        try:
            logger.info("🔄 Forcing complete CUDA context reset...")

            # 1. Clear all cached memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

                # 2. Reset memory stats
                if hasattr(torch.cuda, "reset_peak_memory_stats"):
                    torch.cuda.reset_peak_memory_stats()
                if hasattr(torch.cuda, "reset_max_memory_allocated"):
                    torch.cuda.reset_max_memory_allocated()

            # 3. Force garbage collection multiple times
            for _ in range(3):
                gc.collect()
                time.sleep(0.5)

            # 4. Reset CUDA context (nuclear option)
            if torch.cuda.is_available():
                try:
                    # This forces complete CUDA context reset
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                    torch.cuda.synchronize()

                    # Reinitialize CUDA
                    torch.cuda.init()
                    logger.info("✅ CUDA context reset successful")

                except Exception as e:
                    logger.warning(f"CUDA context reset partial failure: {e}")

            # 5. Additional cleanup
            if hasattr(torch, "_C") and hasattr(
                torch._C, "_cuda_clearCublasWorkspaces"
            ):
                torch._C._cuda_clearCublasWorkspaces()

            logger.info("✅ Complete CUDA reset finished")

        except Exception as e:
            logger.error(f"❌ CUDA reset failed: {e}")

    def _disable_auto_continual_on_failure(self):
        """Disable auto-continual learning when CUDA issues detected"""
        logger.warning("🚨 Disabling auto-continual learning due to CUDA issues")
        logger.warning("🔄 Forcing fresh training to recover from CUDA corruption")
        return True  # Force fresh training

    def _run_enhanced_training(
        self,
        job_id: str,
        model_path: str,
        examples: List[Dict],
        config: Dict,
        base_job_id: str = None,
    ):
        """Enhanced training execution with container restart capability and FIXED memory management"""
        job = self.jobs[job_id]
        self.active_training = True
        # Persist effective config so helper methods can honor user overrides
        job._effective_config = config

        try:
            # FIXED: Use stored instance
            if self.llama_manager and self.llama_manager.is_model_loaded():
                logger.info("Unloading GGUF model to free GPU for training...")
                self.llama_manager.unload_model()
                time.sleep(5)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logger.info(
                    "GPU memory freed. Available: %.1fGB",
                    self._get_available_gpu_memory(),
                )
        except Exception as e:
            logger.warning(f"Could not unload GGUF: {e}")

        try:
            job.status = "running"
            self._save_jobs_to_disk()

            # CONTINUAL LEARNING WITH IMPROVED FALLBACK STRATEGY
            if base_job_id:
                auto_msg = " (AUTO-DETECTED)" if job.auto_continued else ""
                job.message = (
                    f"Starting continual training from job {base_job_id}{auto_msg}..."
                )
                logger.info(
                    f"Starting continual training for job {job_id} from base {base_job_id}{auto_msg}"
                )

                # Enhanced pre-cleanup for continual learning
                logger.info("Enhanced pre-clearing memory for continual learning...")
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                time.sleep(3)

                # TRY CONTINUAL LEARNING WITH IMPROVED FALLBACK
                success = self._attempt_continual_learning_with_enhanced_fallback(
                    job, model_path, base_job_id
                )

                if not success:
                    # CRITICAL: DO NOT FALL BACK TO FRESH TRAINING
                    # This would cause data loss!
                    job.status = "failed"
                    job.message = "Continual learning failed - all fallback attempts exhausted. Preserving previous training data."
                    job.end_time = datetime.now()
                    self._save_jobs_to_disk()
                    logger.error(
                        "CONTINUAL LEARNING FAILED - DATA PRESERVED (no fallback to fresh training)"
                    )
                    return

            else:
                # Fresh training
                job.message = "Starting fresh training..."
                logger.info(f"Starting fresh training for job {job_id}")

                success = self._attempt_fresh_training(job, model_path)

                if not success:
                    job.status = "failed"
                    job.message = "Fresh training failed"
                    job.end_time = datetime.now()
                    self._save_jobs_to_disk()
                    return

            # Continue with training process...
            model, tokenizer = job._model, job._tokenizer  # Set by attempt methods

            # Data preparation
            job.progress = 10.0
            job.message = "Preparing training data..."

            if len(examples) >= 10:
                train_examples, val_examples = train_test_split(
                    examples,
                    test_size=0.2,
                    random_state=42,
                    stratify=[ex.get("mapping_type", "general") for ex in examples],
                )
            else:
                train_examples = examples
                val_examples = []

            training_data = self._prepare_enhanced_training_data(train_examples)
            validation_data = (
                self._prepare_enhanced_training_data(val_examples)
                if val_examples
                else []
            )

            # Dataset preparation
            job.progress = 60.0
            job.message = "Tokenizing datasets..."

            train_dataset = Dataset.from_list(training_data)
            val_dataset = (
                Dataset.from_list(validation_data) if validation_data else None
            )

            # Respect configurable sequence length while clamping for stability
            max_seq_length = config.get("max_length")
            if max_seq_length is None:
                max_seq_length = config.get("sequence_length", 1024)
            max_seq_length = int(max_seq_length)
            max_seq_length = max(256, min(max_seq_length, 2048))
            pad_to_multiple = 16 if max_seq_length >= 1024 else 8

            def tokenize_function(examples):
                if isinstance(examples["text"], list):
                    texts = examples["text"]
                else:
                    texts = [examples["text"]]

                result = tokenizer(
                    texts,
                    truncation=True,
                    max_length=max_seq_length,
                    padding=True,
                    return_tensors=None,
                )

                if isinstance(result["input_ids"][0], list):
                    result["labels"] = [ids[:] for ids in result["input_ids"]]
                else:
                    result["labels"] = result["input_ids"][:]

                return result

            train_tokenized = train_dataset.map(
                tokenize_function,
                batched=True,
                remove_columns=["text"],
                desc="Tokenizing training data",
            )

            val_tokenized = None
            if val_dataset:
                val_tokenized = val_dataset.map(
                    tokenize_function,
                    batched=True,
                    remove_columns=["text"],
                    desc="Tokenizing validation data",
                )

            # Training arguments with enhanced memory settings
            job.total_epochs = config.get("epochs", 3)
            warmup_steps = int(len(train_tokenized) * config.get("warmup_ratio", 0.05))

            # Memory-aware batch strategy with sane fallbacks
            per_device_batch_size = max(
                1, int(config.get("per_device_batch_size", 1))
            )
            gradient_accumulation_steps = max(
                1,
                int(
                    config.get(
                        "gradient_accumulation_steps",
                        max(1, config.get("batch_size", 1)),
                    )
                ),
            )
            effective_batch = per_device_batch_size * gradient_accumulation_steps

            bf16_available = (
                self.cuda_available
                and hasattr(torch.cuda, "is_bf16_supported")
                and torch.cuda.is_bf16_supported()
            )
            fp16_enabled = self.cuda_available and not bf16_available

            training_args = TrainingArguments(
                output_dir=job.output_path,
                num_train_epochs=job.total_epochs,
                per_device_train_batch_size=per_device_batch_size,
                per_device_eval_batch_size=per_device_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                learning_rate=config.get("learning_rate", 2e-4),
                weight_decay=0.01,
                warmup_steps=warmup_steps,
                logging_steps=max(1, len(train_tokenized) // max(10, effective_batch)),
                eval_steps=max(10, len(train_tokenized) // 10)
                if val_tokenized
                else None,
                save_steps=max(50, len(train_tokenized) // 4),
                eval_strategy="steps" if val_tokenized else "no",
                save_strategy="steps",
                load_best_model_at_end=True if val_tokenized else False,
                metric_for_best_model="eval_loss" if val_tokenized else None,
                greater_is_better=False,
                bf16=bf16_available,
                fp16=fp16_enabled,
                gradient_checkpointing=self.cuda_available,
                dataloader_pin_memory=False,  # Disable for memory conservation
                remove_unused_columns=False,
                report_to=None,
                optim="adamw_torch",
                lr_scheduler_type="cosine",
                save_total_limit=1,  # Reduced from 2 for memory conservation
                max_grad_norm=1.0,
                # NEW: Additional memory conservation settings
                dataloader_num_workers=0,  # Disable multiprocessing
                max_steps=-1,
                prediction_loss_only=True,
                skip_memory_metrics=True,
            )

            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=False,
                pad_to_multiple_of=pad_to_multiple,
            )

            # Setup callbacks
            callbacks = [EnhancedProgressCallback(job, self)]
            if val_tokenized:
                callbacks.append(
                    EarlyStoppingCallback(
                        early_stopping_patience=3, early_stopping_threshold=0.001
                    )
                )

            # Create trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_tokenized,
                eval_dataset=val_tokenized,
                data_collator=data_collator,
                callbacks=callbacks,
                tokenizer=tokenizer,
            )

            job.progress = 70.0
            training_type_msg = (
                "auto-continual"
                if (base_job_id and job.auto_continued)
                else ("continual" if base_job_id else "fresh")
            )
            job.message = f"Starting {training_type_msg} training..."

            logger.info(
                "Training configuration: batch_size=%s, grad_accum=%s, epochs=%s, max_seq_len=%s",
                per_device_batch_size,
                gradient_accumulation_steps,
                job.total_epochs,
                max_seq_length,
            )

            # Execute training
            train_result = trainer.train()
            logger.info("Training completed successfully")

            # Save model
            job.progress = 95.0
            job.message = "Saving trained model..."

            trainer.save_model()
            tokenizer.save_pretrained(job.output_path)

            # Save training metrics
            metrics_file = f"{job.output_path}/training_metrics.json"
            with open(metrics_file, "w") as f:
                json.dump(
                    {
                        "training_metrics": job.training_metrics,
                        "validation_metrics": job.validation_metrics,
                        "final_train_loss": train_result.training_loss,
                        "data_analysis": job.data_analysis,
                        "config_used": config,
                        "training_type": job.training_type,
                        "base_job_id": job.base_job_id,
                        "model_version": job.model_version,
                        "auto_continued": job.auto_continued,
                        "cuda_available": self.cuda_available,
                    },
                    f,
                    indent=2,
                )

            # IMPROVED: Store trained model for inference with memory limit check
            try:
                if len(self.trained_models) < self.max_models_in_memory:
                    self.trained_models[job_id] = (model, tokenizer)
                    logger.info("Model stored in memory for inference")
                else:
                    # Remove oldest model to make space
                    if self.trained_models:
                        oldest_job_id = list(self.trained_models.keys())[0]
                        old_model, old_tokenizer = self.trained_models[oldest_job_id]
                        if hasattr(old_model, "cpu"):
                            old_model.cpu()
                        del old_model, old_tokenizer
                        del self.trained_models[oldest_job_id]
                        gc.collect()

                    self.trained_models[job_id] = (model, tokenizer)
                    logger.info("Model stored in memory (replaced oldest model)")

            except Exception as memory_store_error:
                logger.warning(f"Could not store model in memory: {memory_store_error}")

            # Success
            job.status = "completed"
            job.progress = 100.0
            training_type_msg = (
                "auto-continual" if job.auto_continued else job.training_type
            )
            job.message = f"{training_type_msg} training completed successfully. Final loss: {train_result.training_loss:.4f}"
            job.end_time = datetime.now()

            self._save_jobs_to_disk()

            logger.info(
                f"✅ {training_type_msg} training completed successfully for job {job_id} (v{job.model_version})"
            )

            # Clean up old models after successful completion
            try:
                self._cleanup_old_models()
            except Exception as cleanup_error:
                logger.warning(f"Model cleanup warning: {cleanup_error}")

            # CHECK FOR CONTAINER RESTART AFTER SUCCESSFUL TRAINING
            if self._should_restart_container():
                logger.info(
                    "🔄 Training completed - scheduling container restart for memory recovery"
                )
                self._schedule_container_restart(job_id)
            else:
                logger.info("Training completed - no restart needed")

        except Exception as e:
            job.status = "failed"
            job.message = f"Training failed: {str(e)}"
            job.end_time = datetime.now()
            self._save_jobs_to_disk()
            logger.error(f"Training failed with error: {e}", exc_info=True)

        finally:
            self.active_training = False
            # Enhanced cleanup
            try:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            except:
                pass

            logger.info(f"Training cleanup completed for job {job_id}")

    def _attempt_continual_learning_with_enhanced_fallback(
        self, job: TrainingJob, model_path: str, base_job_id: str
    ) -> bool:
        """Attempt continual learning with ENHANCED fallback strategies and better memory management"""

        # Get all valid jobs as fallback options
        valid_jobs = self.get_all_valid_jobs_for_continual_learning()

        if not valid_jobs:
            logger.error("No valid jobs found for continual learning")
            return False

        # Try each valid job in order
        for attempt_num, fallback_job_id in enumerate(valid_jobs):
            logger.info(
                f"Attempting continual learning with job {fallback_job_id} (attempt {attempt_num + 1}/{len(valid_jobs)})"
            )

            try:
                # ENHANCED: Get current available memory and adjust configs accordingly
                available_memory = self._get_available_gpu_memory()
                logger.info(
                    f"Available GPU memory before attempt: {available_memory:.1f} GB"
                )

                # IMPROVED: More conservative memory configurations
                memory_configs = [
                    {
                        "4bit": True,
                        "double_quant": False,
                        "max_gpu": f"{max(6, int(available_memory * 0.6))}GB",
                        "max_cpu": "40GB",
                    },
                    {
                        "4bit": True,
                        "double_quant": False,
                        "max_gpu": f"{max(4, int(available_memory * 0.4))}GB",
                        "max_cpu": "44GB",
                    },
                    {
                        "8bit": True,
                        "max_gpu": f"{max(3, int(available_memory * 0.3))}GB",
                        "max_cpu": "48GB",
                    },
                    {"cpu_fallback": True},
                ]

                for config_num, mem_config in enumerate(memory_configs):
                    logger.info(f"  Memory config {config_num + 1}: {mem_config}")

                    # Apply dynamic adjustment based on available memory
                    adjusted_config = self._adjust_memory_config_for_continual(
                        mem_config
                    )

                    success = self._load_continual_model(
                        job, model_path, fallback_job_id, adjusted_config
                    )

                    if success:
                        logger.info(
                            f"✅ Continual learning setup successful with job {fallback_job_id}"
                        )
                        # Update job to reflect the successful base job
                        job.base_job_id = fallback_job_id
                        if fallback_job_id != base_job_id:
                            logger.info(
                                f"Updated base job from {base_job_id} to {fallback_job_id}"
                            )
                        return True
                    else:
                        # Force cleanup between attempts
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize()
                        time.sleep(1)

                logger.warning(f"All memory configs failed for job {fallback_job_id}")

            except Exception as e:
                logger.warning(f"Fallback job {fallback_job_id} failed: {e}")
                # Force cleanup on error
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue

        logger.error("All continual learning fallback attempts failed")
        return False

    def _load_continual_model(
        self, job: TrainingJob, model_path: str, base_job_id: str, mem_config: dict
    ) -> bool:
        """Load model for continual learning with FIXED gradient configuration"""

        try:
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    torch.cuda.init()  # Reinitialize CUDA
                    # Reset NVML
                    import subprocess
                    subprocess.run(['nvidia-smi'], capture_output=True, timeout=5)
                except Exception as cuda_init_error:
                    logger.warning(f"CUDA reinitialization warning: {cuda_init_error}")
            
            # Load tokenizer first (minimal memory impact)
            tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.pad_token_id = tokenizer.eos_token_id

            # Load base model with improved memory configuration
            bnb_config = None
            if mem_config.get("cpu_fallback"):
                # CPU fallback
                model = self._load_llama_with_fallback(
                    model_path,
                    device_map="cpu",
                    torch_dtype=torch.float32,
                    use_cache=False,
                    low_cpu_mem_usage=True,
                )
                self.cuda_available = False

            else:
                # GPU with quantization - IMPROVED SETTINGS
                if mem_config.get("4bit"):
                    bnb_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_use_double_quant=mem_config.get("double_quant", False),
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.bfloat16,
                    )
                elif mem_config.get("8bit"):
                    bnb_config = BitsAndBytesConfig(
                        load_in_8bit=True,
                        llm_int8_threshold=6.0,
                    )
                # IMPROVED: More conservative memory allocation
                max_memory = {
                    0: mem_config.get("max_gpu", "6GB"),
                    "cpu": mem_config.get("max_cpu", "40GB"),
                }

                model = self._load_llama_with_fallback(
                    model_path,
                    quantization_config=bnb_config,
                    device_map="auto",
                    max_memory=max_memory,
                    torch_dtype=torch.bfloat16,
                    use_cache=False,
                    low_cpu_mem_usage=True,
                )

            # CRITICAL: Prepare model for training if using quantization
            if self.cuda_available and bnb_config:
                model = prepare_model_for_kbit_training(
                    model, use_gradient_checkpointing=True
                )

            # CRITICAL: Force cleanup before loading adapter
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            # Load adapter with verification
            base_adapter_path = f"{self.ai_model_dir}/trained_models/{base_job_id}"

            # Verify adapter files
            required_files = ["adapter_config.json", "adapter_model.safetensors"]
            for file in required_files:
                if not os.path.exists(os.path.join(base_adapter_path, file)):
                    raise Exception(f"Missing adapter file: {file}")

            # Load adapter with improved error handling
            model = PeftModel.from_pretrained(
                model,
                base_adapter_path,
                torch_dtype=torch.bfloat16 if self.cuda_available else torch.float32,
                is_trainable=True,
            )

            # CRITICAL FIX: Explicitly enable gradients for adapter parameters
            logger.info("Enabling gradients for continual learning...")

            # Enable gradients for all PEFT parameters
            for name, param in model.named_parameters():
                if any(peft_key in name for peft_key in ["lora_", "adapter_"]):
                    param.requires_grad_(True)
                    logger.debug(f"Enabled gradient for parameter: {name}")

            # Alternative approach: Enable training mode and check gradient requirements
            model.train()

            # Verify that some parameters have gradients enabled
            trainable_params = sum(
                p.numel() for p in model.parameters() if p.requires_grad
            )
            total_params = sum(p.numel() for p in model.parameters())

            logger.info(
                f"Trainable parameters: {trainable_params:,} / {total_params:,} "
                f"({100 * trainable_params / total_params:.2f}%)"
            )

            if trainable_params == 0:
                raise Exception(
                    "No trainable parameters found after loading adapter - gradient setup failed"
                )

            # Additional fix: Ensure adapter modules are in training mode
            for module in model.modules():
                if hasattr(module, "training"):
                    module.train()

            # Final cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Store model and tokenizer in job
            job._model = model
            job._tokenizer = tokenizer

            logger.info(
                "Continual learning model loaded successfully with gradients enabled"
            )
            return True

        except Exception as e:
            logger.warning(f"Continual learning model loading failed: {e}")
            # Enhanced cleanup on failure
            try:
                if "model" in locals():
                    if hasattr(model, "cpu"):
                        model.cpu()
                    del model
                if "tokenizer" in locals():
                    del tokenizer
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            except:
                pass
            return False

    def _attempt_fresh_training(self, job: TrainingJob, model_path: str) -> bool:
        """Attempt fresh training setup with improved memory management"""

        try:
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.pad_token_id = tokenizer.eos_token_id

            # Load base model with improved memory settings
            if self.cuda_available:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                )

                # MORE CONSERVATIVE MEMORY ALLOCATION FOR FRESH TRAINING
                available_memory = self._get_available_gpu_memory()
                safe_gpu_allocation = min(max(6, int(available_memory * 0.7)), 18)
                max_gpu_memory = f"{safe_gpu_allocation}GB"
                max_memory = {0: max_gpu_memory, "cpu": "28GB"}

                model = self._load_llama_with_fallback(
                    model_path,
                    quantization_config=bnb_config,
                    device_map="auto",
                    max_memory=max_memory,
                    torch_dtype=torch.float16,
                    use_cache=False,
                    low_cpu_mem_usage=True,
                )
            else:
                model = self._load_llama_with_fallback(
                    model_path,
                    device_map="cpu",
                    torch_dtype=torch.float32,
                    use_cache=False,
                    low_cpu_mem_usage=True,
                )

            # Setup LoRA
            if self.cuda_available:
                model = prepare_model_for_kbit_training(
                    model, use_gradient_checkpointing=True
                )
            effective_config = getattr(job, "_effective_config", {}) or {}
            lora_r = max(4, min(int(effective_config.get("lora_r", 8)), 16))
            lora_alpha_default = max(lora_r * 2, 16)
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=lora_r,
                lora_alpha=int(
                    effective_config.get("lora_alpha", lora_alpha_default)
                ),
                lora_dropout=float(effective_config.get("lora_dropout", 0.05)),
                target_modules=effective_config.get(
                    "lora_target_modules",
                    ["q_proj", "v_proj", "k_proj", "o_proj"],
                ),
                bias="none",
            )

            model = get_peft_model(model, lora_config)

            # Store model and tokenizer in job
            job._model = model
            job._tokenizer = tokenizer

            logger.info("Fresh training model setup successful")
            return True

        except Exception as e:
            logger.error(f"Fresh training setup failed: {e}")
            # Enhanced cleanup on failure
            try:
                if "model" in locals():
                    if hasattr(model, "cpu"):
                        model.cpu()
                    del model
                if "tokenizer" in locals():
                    del tokenizer
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            except:
                pass
            return False

    def _prepare_enhanced_training_data(self, examples: List[Dict]) -> List[Dict]:
        """Prepare training data with enhanced instructions"""
        training_data = []

        for example in examples:
            instruction = self._get_enhanced_instruction(
                example.get("mapping_type", "general")
            )
            context = self._get_mapping_context(example.get("mapping_type", "general"))

            prompt = f"""<s>[INST] {instruction}

{context}

Input Data:
{json.dumps(example['input_data'], indent=2)}

Convert this to the target format. Follow the mapping rules precisely. [/INST]

{json.dumps(example['output_data'], indent=2)}</s>"""

            training_data.append({"text": prompt})

        return training_data

    # def _get_enhanced_instruction(self, mapping_type: str) -> str:
    #     """Enhanced instructions for different mapping types"""
    #     instructions = {
    #         "otm_to_jt_canonical": (
    #             "You are an expert data transformation specialist for OTM (Oracle Transportation Management) to JT Canonical format conversion. "
    #             "Apply OTM-specific business rules, handle shipment events, transform route data, and ensure proper field mappings. "
    #             "Preserve data integrity and apply conditional transformations based on shipment status and location codes. "
    #             "Map OTM shipment statuses to canonical event codes, transform location codes to standard format, and preserve shipment hierarchy."
    #         ),
    #         "jt_canonical_to_fedex": (
    #             "You are an expert data transformation specialist for JT Canonical to FedEx API format conversion. "
    #             "Apply FedEx-specific formatting rules, transform tracking events, handle delivery status codes, and ensure proper address formatting. "
    #             "Convert canonical event codes to FedEx event codes and maintain shipment timeline accuracy. "
    #             "Follow FedEx API schema requirements and apply proper service type mappings."
    #         ),
    #         "general": (
    #             "You are an expert data transformation specialist. "
    #             "Analyze the input structure, identify the appropriate mapping rules, and transform to the target schema. "
    #             "Maintain data accuracy and apply appropriate business logic for the detected format."
    #         ),
    #     }
    #     return instructions.get(mapping_type, instructions["general"])


    # def _get_mapping_context(self, mapping_type: str) -> str:
    #     """Provide mapping-specific context and rules"""
    #     contexts = {
    #         "otm_to_jt_canonical": (
    #             "Key OTM to JT Canonical Rules:\n"
    #             "- Map OTM shipment statuses to canonical event codes (e.g., 'Shipped' -> 'P1', 'Delivered' -> 'DLV')\n"
    #             "- Transform OTM location codes to standard format and preserve routing information\n"
    #             "- Convert OTM timestamps to ISO format and maintain timezone consistency\n"
    #             "- Preserve shipment hierarchy, reference numbers, and carrier information\n"
    #             "- Apply conditional transformations based on shipment type and status"
    #         ),
    #         "jt_canonical_to_fedex": (
    #             "Key JT Canonical to FedEx Rules:\n"
    #             "- Convert canonical event codes to FedEx tracking codes (e.g., 'P1' -> 'DP', 'DLV' -> 'DL')\n"
    #             "- Format addresses according to FedEx API requirements (standardized format)\n"
    #             "- Transform delivery status and service types to FedEx nomenclature\n"
    #             "- Maintain tracking number format and reference consistency\n"
    #             "- Apply FedEx-specific field mappings and validation rules"
    #         ),
    #         "general": (
    #             "General Mapping Rules:\n"
    #             "- Analyze input structure to determine mapping type automatically\n"
    #             "- Apply appropriate transformation rules based on detected format\n"
    #             "- Maintain data consistency and proper validation throughout\n"
    #             "- Handle edge cases and missing data appropriately\n"
    #             "- Preserve business logic and data relationships"
    #         ),
    #     }
    #     return contexts.get(mapping_type, contexts["general"])


    def _get_enhanced_instruction(self, mapping_type: str) -> str:
        """Enhanced instructions for different mapping types"""
        instructions = {
            "otm_to_jt_canonical": (
                "You are an expert data transformation specialist for OTM (Oracle Transportation Management) to JT Canonical format conversion. "
                "Apply OTM-specific business rules, handle shipment events, transform route data, and ensure proper field mappings. "
                "Preserve data integrity and apply conditional transformations based on shipment status and location codes. "
                "Map OTM shipment statuses to canonical event codes, transform location codes to standard format, and preserve shipment hierarchy."
            ),
            "jt_canonical_to_fedex": (
                "You are an expert data transformation specialist for JT Canonical to FedEx API format conversion. "
                "Apply FedEx-specific formatting rules, transform tracking events, handle delivery status codes, and ensure proper address formatting. "
                "Convert canonical event codes to FedEx event codes and maintain shipment timeline accuracy. "
                "Follow FedEx API schema requirements and apply proper service type mappings."
            ),
            "general": (
                "You are an expert data transformation specialist. "
                "Analyze the input structure, identify the appropriate mapping rules, and transform to the target schema. "
                "Maintain data accuracy and apply appropriate business logic for the detected format."
            ),
        }
        # Return specific instruction if available, otherwise use general with mapping_type context
        if mapping_type in instructions:
            return instructions[mapping_type]
        else:
            return (
                f"You are an expert data transformation specialist for {mapping_type} conversion. "
                "Analyze the input structure, identify the appropriate mapping rules, and transform to the target schema. "
                "Maintain data accuracy and apply appropriate business logic for this specific mapping type."
            )



    def _get_mapping_context(self, mapping_type: str) -> str:
        """Provide mapping-specific context and rules"""
        contexts = {
            "otm_to_jt_canonical": (
                "Key OTM to JT Canonical Rules:\n"
                "- Map OTM shipment statuses to canonical event codes (e.g., 'Shipped' -> 'P1', 'Delivered' -> 'DLV')\n"
                "- Transform OTM location codes to standard format and preserve routing information\n"
                "- Convert OTM timestamps to ISO format and maintain timezone consistency\n"
                "- Preserve shipment hierarchy, reference numbers, and carrier information\n"
                "- Apply conditional transformations based on shipment type and status"
            ),
            "jt_canonical_to_fedex": (
                "Key JT Canonical to FedEx Rules:\n"
                "- Convert canonical event codes to FedEx tracking codes (e.g., 'P1' -> 'DP', 'DLV' -> 'DL')\n"
                "- Format addresses according to FedEx API requirements (standardized format)\n"
                "- Transform delivery status and service types to FedEx nomenclature\n"
                "- Maintain tracking number format and reference consistency\n"
                "- Apply FedEx-specific field mappings and validation rules"
            ),
            "general": (
                "General Mapping Rules:\n"
                "- Analyze input structure to determine mapping type automatically\n"
                "- Apply appropriate transformation rules based on detected format\n"
                "- Maintain data consistency and proper validation throughout\n"
                "- Handle edge cases and missing data appropriately\n"
                "- Preserve business logic and data relationships"
            ),
        }
        # Return specific context if available, otherwise use generic context
        if mapping_type in contexts:
            return contexts[mapping_type]
        else:
            return (
                f"Mapping Rules for {mapping_type}:\n"
                "- Analyze input structure to determine appropriate transformation rules\n"
                "- Apply format-specific business logic and validation\n"
                "- Maintain data consistency and proper field mappings\n"
                "- Handle edge cases and preserve data relationships\n"
                "- Follow best practices for this specific mapping type"
            )


    def predict_with_trained_model(
        self, job_id: str, input_data: Dict, mapping_type: str
    ) -> Dict:
        """Enhanced prediction with trained model"""
        if job_id not in self.trained_models:
            return {"error": "Trained model not found or not loaded"}

        try:
            model, tokenizer = self.trained_models[job_id]

            instruction = self._get_enhanced_instruction(mapping_type)
            context = self._get_mapping_context(mapping_type)

            prompt = f"""<s>[INST] {instruction}

{context}

Input Data:
{json.dumps(input_data, indent=2)}

Convert this to the target format. Follow the mapping rules precisely. [/INST]"""

            inputs = tokenizer(prompt, return_tensors="pt")

            # Move to correct device
            if self.cuda_available and next(model.parameters()).is_cuda:
                inputs = {k: v.cuda() for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=2048,
                    temperature=0.1,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.05,
                    num_return_sequences=1,
                )

            response = tokenizer.decode(outputs[0], skip_special_tokens=True)

            if "[/INST]" in response:
                generated_mapping = response.split("[/INST]")[-1].strip()

                try:
                    result = json.loads(generated_mapping)
                    return {
                        "success": True,
                        "mapped_data": result,
                        "raw_response": generated_mapping,
                        "mapping_type": mapping_type,
                        "confidence": self._calculate_confidence(generated_mapping),
                    }
                except json.JSONDecodeError as je:
                    try:
                        import re

                        json_match = re.search(r"\{.*\}", generated_mapping, re.DOTALL)
                        if json_match:
                            partial_json = json_match.group()
                            result = json.loads(partial_json)
                            return {
                                "success": True,
                                "mapped_data": result,
                                "raw_response": generated_mapping,
                                "mapping_type": mapping_type,
                                "warning": "Extracted partial JSON from response",
                            }
                    except:
                        pass

                    return {
                        "success": False,
                        "mapped_data": None,
                        "raw_response": generated_mapping,
                        "error": f"Generated response is not valid JSON: {str(je)}",
                        "mapping_type": mapping_type,
                    }

            return {
                "success": False,
                "mapped_data": None,
                "raw_response": response,
                "error": "Could not extract mapping from response",
                "mapping_type": mapping_type,
            }

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {
                "error": f"Prediction failed: {str(e)}",
                "mapping_type": mapping_type,
            }

    def _calculate_confidence(self, generated_text: str) -> float:
        """Calculate confidence score for generated mapping"""
        confidence = 1.0

        if len(generated_text) < 50:
            confidence *= 0.5

        if "error" in generated_text.lower() or "invalid" in generated_text.lower():
            confidence *= 0.3

        try:
            json.loads(generated_text)
            confidence *= 1.2
        except:
            confidence *= 0.7

        return min(confidence, 1.0)

    def get_current_best_model(self) -> Optional[str]:
        """Get the current best model (most recent successful job) for predictions"""
        return self.get_latest_successful_job()

    def get_disk_usage_info(self) -> Dict:
        """Get disk usage information for model directories"""
        try:
            models_dir = f"{self.ai_model_dir}/trained_models"

            if not os.path.exists(models_dir):
                return {"error": "Models directory not found"}

            # Calculate total size
            total_size = 0
            model_sizes = {}

            for job_id in os.listdir(models_dir):
                job_dir = os.path.join(models_dir, job_id)
                if os.path.isdir(job_dir):
                    job_size = sum(
                        os.path.getsize(os.path.join(dirpath, filename))
                        for dirpath, dirnames, filenames in os.walk(job_dir)
                        for filename in filenames
                    )
                    model_sizes[job_id] = job_size
                    total_size += job_size

            return {
                "total_models": len(model_sizes),
                "total_size_mb": total_size / 1024**2,
                "total_size_gb": total_size / 1024**3,
                "individual_model_sizes": {
                    job_id: {"size_mb": size / 1024**2, "size_gb": size / 1024**3}
                    for job_id, size in model_sizes.items()
                },
                "max_models_to_keep": self.max_models_to_keep,
                "max_models_in_memory": self.max_models_in_memory,
                "models_currently_in_memory": len(self.trained_models),
            }

        except Exception as e:
            return {"error": f"Failed to calculate disk usage: {str(e)}"}

    def force_cleanup_models(self, keep_count: int = None) -> Dict:
        """Force cleanup of old models with custom keep count"""
        try:
            if keep_count is not None:
                original_max = self.max_models_to_keep
                self.max_models_to_keep = keep_count

            # Also cleanup memory
            self._unload_models_from_memory()

            self._cleanup_old_models()

            if keep_count is not None:
                self.max_models_to_keep = original_max

            return {
                "success": True,
                "message": f"Model cleanup completed, keeping {keep_count or self.max_models_to_keep} models",
            }

        except Exception as e:
            return {"success": False, "error": f"Cleanup failed: {str(e)}"}

    def get_system_status(self) -> Dict:
        """Enhanced system status with memory management metrics"""
        return {
            "training_available": True,
            "active_training": self.active_training,
            "total_jobs": len(self.jobs),
            "completed_jobs": len(
                [j for j in self.jobs.values() if j.status == "completed"]
            ),
            "failed_jobs": len([j for j in self.jobs.values() if j.status == "failed"]),
            "running_jobs": len(
                [j for j in self.jobs.values() if j.status == "running"]
            ),
            "trained_models_loaded": len(self.trained_models),
            "max_models_in_memory": self.max_models_in_memory,
            "auto_continual_jobs": len(
                [j for j in self.jobs.values() if j.auto_continued]
            ),
            "manual_continual_jobs": len(
                [
                    j
                    for j in self.jobs.values()
                    if j.training_type == "continual" and not j.auto_continued
                ]
            ),
            "fresh_jobs": len(
                [j for j in self.jobs.values() if j.training_type == "fresh"]
            ),
            "current_best_model": self.get_current_best_model(),
            "gpu_available": self.cuda_available,
            "available_gpu_memory_gb": self._get_available_gpu_memory(),
            "cuda_error": self.cuda_error,
            "cuda_device_count": torch.cuda.device_count()
            if torch.cuda.is_available()
            else 0,
            "gpu_memory_allocated": torch.cuda.memory_allocated()
            if torch.cuda.is_available()
            else 0,
            "gpu_memory_cached": torch.cuda.memory_reserved()
            if torch.cuda.is_available()
            else 0,
            "ai_model_dir": self.ai_model_dir,
            "jobs_persisted": os.path.exists(self.jobs_file),
            "valid_continual_models": len(
                self.get_all_valid_jobs_for_continual_learning()
            ),
            "enhancement_features": [
                "auto_continual_learning",
                "data_loss_prevention",
                "intelligent_fallback",
                "persistent_job_storage",
                "automatic_model_cleanup",
                "robust_error_recovery",
                "multi_tier_memory_fallback",
                "adaptive_quantization",
                "cuda_error_recovery",
                "enhanced_memory_management",
                "conservative_memory_allocation",
                "dynamic_memory_adjustment",
            ],
        }
