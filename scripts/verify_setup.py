#!/usr/bin/env python3
"""
Setup Verification Script
Verifies that all components are properly configured before starting the application
"""

import os
import sys
import sqlite3
import logging
from pathlib import Path

# Add the app directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'app'))

def setup_logging():
    """Setup logging for verification"""
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        handlers=[logging.StreamHandler()]
    )
    return logging.getLogger(__name__)

def check_python_version():
    """Check if Python version is compatible"""
    logger = logging.getLogger(__name__)
    
    version = sys.version_info
    if version.major != 3 or version.minor < 8:
        logger.error(f"Python 3.8+ required, found {version.major}.{version.minor}")
        return False
    
    logger.info(f"✓ Python version: {version.major}.{version.minor}.{version.micro}")
    return True

def check_dependencies():
    """Check if all required dependencies are installed"""
    logger = logging.getLogger(__name__)
    
    required_packages = [
        ('flask', 'Flask'),
        ('llama_cpp', 'llama-cpp-python'),
        ('sqlite3', 'sqlite3 (built-in)'),
    ]
    
    missing_packages = []
    
    for package, name in required_packages:
        try:
            __import__(package)
            logger.info(f"✓ {name} is available")
        except ImportError:
            logger.error(f"✗ {name} is missing")
            missing_packages.append(name)
    
    if missing_packages:
        logger.error(f"Missing packages: {', '.join(missing_packages)}")
        return False
    
    return True

def check_cuda_support():
    """Check CUDA support and availability"""
    logger = logging.getLogger(__name__)
    
    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(current_device)
            logger.info(f"✓ CUDA available: {device_count} device(s)")
            logger.info(f"  Current device: {device_name}")
            return True
        else:
            logger.warning("⚠ CUDA not available, will use CPU")
            return True  # Not an error, just a warning
    except ImportError:
        logger.warning("⚠ PyTorch not installed, cannot check CUDA")
        return True  # Not critical for basic functionality

def check_model_file():
    """Check if the model file exists and is accessible"""
    logger = logging.getLogger(__name__)
    
    model_path = os.environ.get('MODEL_PATH', '/app/ai-model/codellama-13b.Q4_K_M.gguf')
    
    if not os.path.exists(model_path):
        logger.error(f"✗ Model file not found: {model_path}")
        return False
    
    # Check file size (should be several GB for LLaMA model)
    file_size = os.path.getsize(model_path)
    file_size_gb = file_size / (1024**3)
    
    if file_size_gb < 1:
        logger.warning(f"⚠ Model file seems small: {file_size_gb:.2f} GB")
    else:
        logger.info(f"✓ Model file found: {model_path} ({file_size_gb:.2f} GB)")
    
    # Check read permissions
    if not os.access(model_path, os.R_OK):
        logger.error(f"✗ Cannot read model file: {model_path}")
        return False
    
    return True

def check_directories():
    """Check and create necessary directories"""
    logger = logging.getLogger(__name__)
    
    required_dirs = [
        '/app/logs',
        '/app/data',
        '/app/temp',
        '/app/.cache'
    ]
    
    for directory in required_dirs:
        try:
            os.makedirs(directory, exist_ok=True)
            if os.access(directory, os.W_OK):
                logger.info(f"✓ Directory ready: {directory}")
            else:
                logger.error(f"✗ Directory not writable: {directory}")
                return False
        except Exception as e:
            logger.error(f"✗ Cannot create directory {directory}: {str(e)}")
            return False
    
    return True

def check_database():
    """Test database connectivity and basic operations"""
    logger = logging.getLogger(__name__)
    
    try:
        # Test SQLite database
        db_path = os.environ.get('DATABASE_PATH', '/app/data/users.db')
        
        # Ensure database directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Test database connection
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT sqlite_version()")
            version = cursor.fetchone()[0]
            logger.info(f"✓ SQLite database ready (version: {version})")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Database check failed: {str(e)}")
        return False

def check_environment_variables():
    """Check required environment variables"""
    logger = logging.getLogger(__name__)
    
    # Required environment variables
    required_vars = {
        'MODEL_PATH': '/app/ai-model/codellama-13b.Q4_K_M.gguf',
        'USE_CUDA': 'true',
        'LOG_LEVEL': 'INFO'
    }
    
    # Optional environment variables
    optional_vars = {
        'PORT': '8000',
        'HOST': '0.0.0.0',
        'DATABASE_PATH': '/app/data/users.db'
    }
    
    all_good = True
    
    # Check required variables
    for var, default in required_vars.items():
        value = os.environ.get(var, default)
        if value:
            logger.info(f"✓ {var}: {value}")
        else:
            logger.error(f"✗ Missing required environment variable: {var}")
            all_good = False
    
    # Check optional variables
    for var, default in optional_vars.items():
        value = os.environ.get(var, default)
        logger.info(f"  {var}: {value}")
    
    return all_good

def main():
    """Run all verification checks"""
    logger = setup_logging()
    
    logger.info("=" * 50)
    logger.info("Starting Application Setup Verification")
    logger.info("=" * 50)
    
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("CUDA Support", check_cuda_support),
        ("Model File", check_model_file),
        ("Directories", check_directories),
        ("Database", check_database),
        ("Environment Variables", check_environment_variables)
    ]
    
    failed_checks = []
    
    for check_name, check_func in checks:
        logger.info(f"\n--- {check_name} ---")
        try:
            if not check_func():
                failed_checks.append(check_name)
        except Exception as e:
            logger.error(f"✗ {check_name} check failed with error: {str(e)}")
            failed_checks.append(check_name)
    
    logger.info("\n" + "=" * 50)
    if failed_checks:
        logger.error(f"Verification FAILED! Failed checks: {', '.join(failed_checks)}")
        logger.error("Please fix the above issues before starting the application.")
        return 1
    else:
        logger.info("✓ All verification checks PASSED!")
        logger.info("Application is ready to start.")
        return 0

if __name__ == "__main__":
    sys.exit(main())