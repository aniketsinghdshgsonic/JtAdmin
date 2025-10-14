






# app/services/__init__.py
"""
Services Package
Contains business logic and service layer classes
"""

import logging
logger = logging.getLogger(__name__)


MappingStorageService = None
HybridTemplateEngine = None
TemplateReconstructor = None
FocusedFieldMapper = None
LLMTrainingManager = None
InteractiveChatService = None
FieldAnalyzer = None
SmartHybridOrchestrator = None


try:
    from .mapping_storage_service import MappingStorageService  
    from .hybrid_template_engine import HybridTemplateEngine
    from .template_reconstructor import TemplateReconstructor
    from .field_mapper import FocusedFieldMapper
    from .llm_training_manager import LLMTrainingManager
    from .interactive_chat_service import InteractiveChatService
    from .field_analyzer import FieldAnalyzer
    from .smart_hybrid_orchestrator import SmartHybridOrchestrator
    logger.info("✓ MappingStorageService imported successfully")
except ImportError as e:
    logger.error(f"❌ Failed to import MappingStorageService: {e}")
    logger.error("This likely means sentence_transformers is not properly installed")
    MappingStorageService = None
    HybridTemplateEngine = None
    TemplateReconstructor = None
    FocusedFieldMapper = None
    LLMTrainingManager = None
    InteractiveChatService = None
    FieldAnalyzer = None
    SmartHybridOrchestrator = None

# Export what's available
available_services = []
if MappingStorageService:
    available_services.append('MappingStorageService')
if HybridTemplateEngine:
    available_services.append('HybridTemplateEngine')
if TemplateReconstructor:
    available_services.append('TemplateReconstructor')
if FocusedFieldMapper:
    available_services.append('FocusedFieldMapper')
if LLMTrainingManager:
    available_services.append('LLMTrainingManager')
if InteractiveChatService:
    available_services.append('InteractuveChatService')
if FieldAnalyzer:
    available_services.append('FieldAnalyzer')
if SmartHybridOrchestrator:
    available_services.append('SmartHybridOrchestrator')

__all__ = available_services

logger.info(f"Services initialized: {available_services}")














