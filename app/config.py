










# pratik and aniket# app/config.py
import os
import logging
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class Config:
    """Configuration management for JSON Mapping System"""
    
    # Application Settings
    SECRET_KEY = os.getenv("SECRET_KEY", "json-mapping-secret-key-change-in-production")
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", "8000"))
    
    # Environment
    ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
    
    # Flask Settings
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB
    JSON_SORT_KEYS = False
    JSONIFY_PRETTYPRINT_REGULAR = True
    
    # Model Configuration
    MODEL_PATH = os.getenv("MODEL_PATH", "/app/ai-model/codellama-13b.Q4_K_M.gguf")
    MODEL_CONTEXT_SIZE = int(os.getenv("MODEL_CONTEXT_SIZE", "4096"))
    MODEL_MAX_TOKENS = int(os.getenv("MODEL_MAX_TOKENS", "1024"))
    MODEL_TEMPERATURE = float(os.getenv("MODEL_TEMPERATURE", "0.1"))
    USE_CUDA = os.getenv("USE_CUDA", "true").lower() == "true"
    
    # LLM Generation Settings
    DEFAULT_MAX_TOKENS = int(os.getenv("DEFAULT_MAX_TOKENS", "1024"))
    DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", "0.1"))
    DEFAULT_TOP_P = float(os.getenv("DEFAULT_TOP_P", "0.9"))
    DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "40"))
    DEFAULT_REPEAT_PENALTY = float(os.getenv("DEFAULT_REPEAT_PENALTY", "1.1"))
    
    # Generation Timeouts
    GENERATION_TIMEOUT = int(os.getenv("GENERATION_TIMEOUT", "45"))
    MODEL_LOAD_TIMEOUT = int(os.getenv("MODEL_LOAD_TIMEOUT", "300"))
    
    # Milvus Configuration
    MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
    MILVUS_PORT = int(os.getenv("MILVUS_PORT", "19530"))
    MILVUS_USERNAME = os.getenv("MILVUS_USERNAME", "")
    MILVUS_PASSWORD = os.getenv("MILVUS_PASSWORD", "")
    MILVUS_DB_NAME = os.getenv("MILVUS_DB_NAME", "default")
    MILVUS_COLLECTION_NAME = os.getenv("MILVUS_COLLECTION_NAME", "json_mapping_pairs")
    
    # Milvus Index Settings
    MILVUS_INDEX_TYPE = os.getenv("MILVUS_INDEX_TYPE", "IVF_FLAT")
    MILVUS_METRIC_TYPE = os.getenv("MILVUS_METRIC_TYPE", "COSINE")
    MILVUS_NLIST = int(os.getenv("MILVUS_NLIST", "128"))
    MILVUS_NPROBE = int(os.getenv("MILVUS_NPROBE", "10"))
    
    # Embedding Configuration
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "768"))
    EMBEDDING_CACHE_SIZE = int(os.getenv("EMBEDDING_CACHE_SIZE", "1000"))
    
    # Search and Retrieval Settings
    DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "3"))
    MAX_TOP_K = int(os.getenv("MAX_TOP_K", "10"))
    SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.1"))
    
    # Validation Settings
    MIN_CONFIDENCE_SCORE = float(os.getenv("MIN_CONFIDENCE_SCORE", "0.5"))
    HIGH_CONFIDENCE_THRESHOLD = float(os.getenv("HIGH_CONFIDENCE_THRESHOLD", "0.8"))
    FIELD_COVERAGE_THRESHOLD = float(os.getenv("FIELD_COVERAGE_THRESHOLD", "0.7"))
    
    # Processing Settings
    MAX_RETRY_ATTEMPTS = int(os.getenv("MAX_RETRY_ATTEMPTS", "3"))
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "10"))
    CONCURRENT_REQUESTS = int(os.getenv("CONCURRENT_REQUESTS", "5"))
    
    # Logging Configuration
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
    LOG_DIR = os.getenv("LOG_DIR", "/app/logs")
    LOG_FILE = os.getenv("LOG_FILE", "mapping_system.log")
    LOG_MAX_BYTES = int(os.getenv("LOG_MAX_BYTES", "10485760"))  # 10MB
    LOG_BACKUP_COUNT = int(os.getenv("LOG_BACKUP_COUNT", "5"))
    
    # Cache and Storage
    CACHE_DIR = os.getenv("CACHE_DIR", "/app/.cache")
    TEMP_DIR = os.getenv("TEMP_DIR", "/app/temp")
    DATA_DIR = os.getenv("DATA_DIR", "/app/data")
    
    # Performance Settings
    THREAD_POOL_SIZE = int(os.getenv("THREAD_POOL_SIZE", "4"))
    MEMORY_LIMIT_MB = int(os.getenv("MEMORY_LIMIT_MB", "8192"))
    
    # Domain Configuration
    DEFAULT_DOMAIN = os.getenv("DEFAULT_DOMAIN", "general")
    SUPPORTED_DOMAINS = os.getenv("SUPPORTED_DOMAINS", "general,airline,healthcare,finance,ecommerce").split(",")
    
    # System Behavior
    AUTO_STORE_SUCCESSFUL_MAPPINGS = os.getenv("AUTO_STORE_SUCCESSFUL_MAPPINGS", "false").lower() == "true"
    ENABLE_FEEDBACK_LEARNING = os.getenv("ENABLE_FEEDBACK_LEARNING", "true").lower() == "true"
    STRICT_JSON_VALIDATION = os.getenv("STRICT_JSON_VALIDATION", "true").lower() == "true"
    
    # Security Settings
    API_RATE_LIMIT = os.getenv("API_RATE_LIMIT", "100")  # requests per hour
    MAX_REQUEST_SIZE = int(os.getenv("MAX_REQUEST_SIZE", "1048576"))  # 1MB
    ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")
    
    # Development Settings
    DEVELOPMENT_MODE = ENVIRONMENT in ["development", "dev", "local"]
    ENABLE_DETAILED_LOGGING = DEVELOPMENT_MODE or os.getenv("ENABLE_DETAILED_LOGGING", "false").lower() == "true"
    ENABLE_MODEL_TESTING = DEVELOPMENT_MODE or os.getenv("ENABLE_MODEL_TESTING", "false").lower() == "true"
    
    @classmethod
    def validate_config(cls) -> bool:
        """Validate configuration settings"""
        try:
            logger.info("Validating configuration...")
            
            issues = []
            
            # Check required paths
            model_path = Path(cls.MODEL_PATH)
            if not model_path.exists():
                issues.append(f"Model file not found: {cls.MODEL_PATH}")
            elif not model_path.is_file():
                issues.append(f"Model path is not a file: {cls.MODEL_PATH}")
            else:
                # Check model file size
                size_mb = model_path.stat().st_size / (1024 * 1024)
                if size_mb < 100:  # CodeLlama should be at least 100MB
                    issues.append(f"Model file seems too small: {size_mb:.1f}MB")
                else:
                    logger.info(f"Model file found: {size_mb:.1f}MB")
            
            # Create required directories
            for dir_path in [cls.LOG_DIR, cls.CACHE_DIR, cls.TEMP_DIR, cls.DATA_DIR]:
                try:
                    Path(dir_path).mkdir(parents=True, exist_ok=True)
                    logger.debug(f"Directory ensured: {dir_path}")
                except Exception as e:
                    issues.append(f"Cannot create directory {dir_path}: {e}")
            
            # Validate numeric ranges
            if not (512 <= cls.MODEL_CONTEXT_SIZE <= 32768):
                issues.append(f"MODEL_CONTEXT_SIZE should be between 512-32768, got: {cls.MODEL_CONTEXT_SIZE}")
            
            if not (0.0 <= cls.DEFAULT_TEMPERATURE <= 2.0):
                issues.append(f"DEFAULT_TEMPERATURE should be between 0.0-2.0, got: {cls.DEFAULT_TEMPERATURE}")
            
            if not (1 <= cls.DEFAULT_TOP_K <= 100):
                issues.append(f"DEFAULT_TOP_K should be between 1-100, got: {cls.DEFAULT_TOP_K}")
            
            if not (1 <= cls.MILVUS_PORT <= 65535):
                issues.append(f"MILVUS_PORT should be between 1-65535, got: {cls.MILVUS_PORT}")
            
            if not (128 <= cls.EMBEDDING_DIMENSION <= 2048):
                issues.append(f"EMBEDDING_DIMENSION should be between 128-2048, got: {cls.EMBEDDING_DIMENSION}")
            
            # Validate embedding model compatibility
            if cls.EMBEDDING_MODEL == "all-MiniLM-L6-v2" and cls.EMBEDDING_DIMENSION != 768:
                issues.append(f"all-MiniLM-L6-v2 requires dimension 768, got: {cls.EMBEDDING_DIMENSION}")
            
            # Check connectivity requirements
            if cls.MILVUS_HOST in ["localhost", "127.0.0.1"] and os.getenv("DOCKER_ENV"):
                logger.warning("Using localhost for Milvus in Docker environment - this may cause connection issues")
            
            # Validate domains
            if cls.DEFAULT_DOMAIN not in cls.SUPPORTED_DOMAINS:
                issues.append(f"DEFAULT_DOMAIN '{cls.DEFAULT_DOMAIN}' not in SUPPORTED_DOMAINS: {cls.SUPPORTED_DOMAINS}")
            
            # Log configuration summary
            logger.info("Configuration Summary:")
            logger.info(f"  Environment: {cls.ENVIRONMENT}")
            logger.info(f"  Model: {Path(cls.MODEL_PATH).name}")
            logger.info(f"  Context Size: {cls.MODEL_CONTEXT_SIZE}")
            logger.info(f"  CUDA Enabled: {cls.USE_CUDA}")
            logger.info(f"  Milvus: {cls.MILVUS_HOST}:{cls.MILVUS_PORT}")
            logger.info(f"  Embedding Model: {cls.EMBEDDING_MODEL}")
            logger.info(f"  Embedding Dimension: {cls.EMBEDDING_DIMENSION}")
            logger.info(f"  Supported Domains: {cls.SUPPORTED_DOMAINS}")
            logger.info(f"  Log Level: {cls.LOG_LEVEL}")
            
            if issues:
                logger.error("Configuration validation failed:")
                for issue in issues:
                    logger.error(f"  - {issue}")
                return False
            
            logger.info("Configuration validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Configuration validation error: {e}")
            return False
    
    @classmethod
    def get_model_config(cls) -> Dict[str, Any]:
        """Get model-specific configuration"""
        return {
            "model_path": cls.MODEL_PATH,
            "context_size": cls.MODEL_CONTEXT_SIZE,
            "use_cuda": cls.USE_CUDA,
            "generation": {
                "max_tokens": cls.DEFAULT_MAX_TOKENS,
                "temperature": cls.DEFAULT_TEMPERATURE,
                "top_p": cls.DEFAULT_TOP_P,
                "top_k": cls.DEFAULT_TOP_K,
                "repeat_penalty": cls.DEFAULT_REPEAT_PENALTY,
                "timeout": cls.GENERATION_TIMEOUT
            }
        }
    
    @classmethod
    def get_milvus_config(cls) -> Dict[str, Any]:
        """Get Milvus-specific configuration"""
        config = {
            "host": cls.MILVUS_HOST,
            "port": cls.MILVUS_PORT,
            "db_name": cls.MILVUS_DB_NAME,
            "collection_name": cls.MILVUS_COLLECTION_NAME,
            "index": {
                "type": cls.MILVUS_INDEX_TYPE,
                "metric_type": cls.MILVUS_METRIC_TYPE,
                "params": {
                    "nlist": cls.MILVUS_NLIST
                }
            },
            "search": {
                "params": {
                    "nprobe": cls.MILVUS_NPROBE
                }
            }
        }
        
        if cls.MILVUS_USERNAME:
            config["username"] = cls.MILVUS_USERNAME
        if cls.MILVUS_PASSWORD:
            config["password"] = cls.MILVUS_PASSWORD
            
        return config
    
    @classmethod
    def get_embedding_config(cls) -> Dict[str, Any]:
        """Get embedding-specific configuration"""
        return {
            "model_name": cls.EMBEDDING_MODEL,
            "dimension": cls.EMBEDDING_DIMENSION,
            "cache_size": cls.EMBEDDING_CACHE_SIZE
        }
    
    @classmethod
    def get_processing_config(cls) -> Dict[str, Any]:
        """Get processing-specific configuration"""
        return {
            "default_top_k": cls.DEFAULT_TOP_K,
            "max_top_k": cls.MAX_TOP_K,
            "similarity_threshold": cls.SIMILARITY_THRESHOLD,
            "min_confidence": cls.MIN_CONFIDENCE_SCORE,
            "high_confidence": cls.HIGH_CONFIDENCE_THRESHOLD,
            "field_coverage_threshold": cls.FIELD_COVERAGE_THRESHOLD,
            "max_retry_attempts": cls.MAX_RETRY_ATTEMPTS,
            "batch_size": cls.BATCH_SIZE,
            "concurrent_requests": cls.CONCURRENT_REQUESTS
        }
    
    @classmethod
    def is_development(cls) -> bool:
        """Check if running in development mode"""
        return cls.DEVELOPMENT_MODE
    
    @classmethod
    def is_production(cls) -> bool:
        """Check if running in production mode"""
        return cls.ENVIRONMENT == "production"
    
    @classmethod
    def get_cors_config(cls) -> Dict[str, Any]:
        """Get CORS configuration"""
        return {
            "origins": cls.ALLOWED_ORIGINS,
            "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            "headers": ["Content-Type", "Authorization", "X-Requested-With"]
        }
    
    @classmethod
    def get_logging_config(cls) -> Dict[str, Any]:
        """Get logging configuration"""
        return {
            "level": cls.LOG_LEVEL,
            "log_dir": cls.LOG_DIR,
            "log_file": cls.LOG_FILE,
            "max_bytes": cls.LOG_MAX_BYTES,
            "backup_count": cls.LOG_BACKUP_COUNT,
            "detailed": cls.ENABLE_DETAILED_LOGGING
        }

# Environment-specific configurations
class DevelopmentConfig(Config):
    """Development environment configuration"""
    DEBUG = True
    ENABLE_DETAILED_LOGGING = True
    ENABLE_MODEL_TESTING = True
    LOG_LEVEL = "DEBUG"

class ProductionConfig(Config):
    """Production environment configuration"""
    DEBUG = False
    ENABLE_DETAILED_LOGGING = False
    ENABLE_MODEL_TESTING = False
    LOG_LEVEL = "INFO"
    SECRET_KEY = os.getenv("SECRET_KEY")  # Must be set in production
    
    @classmethod
    def validate_config(cls) -> bool:
        """Additional production validation"""
        if not super().validate_config():
            return False
        
        # Production-specific validations
        if cls.SECRET_KEY == "json-mapping-secret-key-change-in-production":
            logger.error("SECRET_KEY must be changed in production")
            return False
        
        if cls.DEBUG:
            logger.warning("DEBUG mode is enabled in production")
        
        return True

class TestingConfig(Config):
    """Testing environment configuration"""
    DEBUG = True
    TESTING = True
    LOG_LEVEL = "DEBUG"
    MODEL_CONTEXT_SIZE = 1024  # Smaller for testing
    DEFAULT_MAX_TOKENS = 256
    GENERATION_TIMEOUT = 15

# Configuration factory
def get_config() -> Config:
    """Get configuration based on environment"""
    env = os.getenv("ENVIRONMENT", "development").lower()
    
    config_map = {
        "development": DevelopmentConfig,
        "dev": DevelopmentConfig,
        "local": DevelopmentConfig,
        "production": ProductionConfig,
        "prod": ProductionConfig,
        "testing": TestingConfig,
        "test": TestingConfig
    }
    
    config_class = config_map.get(env, DevelopmentConfig)
    logger.info(f"Using {config_class.__name__} for environment: {env}")
    
    return config_class