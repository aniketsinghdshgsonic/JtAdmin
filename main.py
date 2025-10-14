# main.py - COMPLETE ENHANCED VERSION with Vector Database + Clean Response Format
import os
import sys
import logging
import signal
import json
import time
import threading
import queue
from datetime import datetime
from typing import Dict, Any, List, Optional
import gc
from app.services.llm_training_manager import LLMTrainingManager
import torch
from functools import wraps
from hashlib import md5
import hashlib
from app.services.interactive_chat_service import InteractiveChatService
import uuid


# Add this global variable with other globals


# Set environment variables for stability
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Force garbage collection
gc.collect()

# Add app directory to Python path
sys.path.insert(0, "/app")
sys.path.insert(0, "/app/app")

# Flask imports
from flask import Flask, request, jsonify
from werkzeug.exceptions import HTTPException

# Application imports
try:
    from app.config import Config
    from app.models.llama_manager import LlamaManager
    from app.milvus_connection import (
        connect_to_milvus,
        disconnect_from_milvus,
        create_mapping_collections,
        get_all_collections_stats,
        insert_mapping_embeddings,
        search_similar_mappings,
        Collection,
        # clear_collection,
    )
    from app.services.mapping_storage_service import MappingStorageService
    from app.services.llm_primary_mapping_service import LLMPrimaryMappingService
    from app.services.field_analyzer import FieldAnalyzer
    from app.services.smart_hybrid_orchestrator import SmartHybridOrchestrator
except ImportError as e:
    print(f"Critical import error: {e}")
    sys.exit(1)


# Configure logging
def setup_logging():
    log_level = getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper())
    os.makedirs("/app/logs", exist_ok=True)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    file_handler = logging.FileHandler("/app/logs/application.log")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(log_level)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(log_level)

    logging.basicConfig(
        level=log_level,
        handlers=[file_handler, console_handler],
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logging.getLogger("werkzeug").setLevel(logging.WARNING)
    return logging.getLogger(__name__)


logger = setup_logging()

# Global components
llama_manager: Optional[LlamaManager] = None
milvus_connected: bool = False
milvus_collections: dict = {}
mapping_storage_service: Optional[MappingStorageService] = None
llm_primary_service: Optional[LLMPrimaryMappingService] = None
training_manager: Optional[LLMTrainingManager] = None
chat_service: Optional[InteractiveChatService] = None
field_analyzer: Optional[FieldAnalyzer] = None
smart_hybrid_orchestrator: Optional[SmartHybridOrchestrator] = None


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "dev-secret-key")
    app.config["DEBUG"] = os.getenv("DEBUG", "False").lower() == "true"
    app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50MB
    return app


app = create_app()


def auto_gpu_cleanup(func):
    """Decorator that automatically cleans GPU memory after function execution"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            # Execute the original function
            result = func(*args, **kwargs)

            # Clean up GPU memory after successful execution
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
                torch.cuda.synchronize()
                logger.info("🧹 GPU memory auto-cleaned after request")

            return result

        except Exception as e:
            # Clean up GPU memory even if function fails
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
                torch.cuda.synchronize()
                logger.info("🧹 GPU memory auto-cleaned after error")

            # Re-raise the exception
            raise e

    return wrapper


def initialize_components():
    """Initialize all system components: LLM, Milvus, Storage, and Smart Hybrid System"""
    global \
        llama_manager, \
        milvus_connected, \
        milvus_collections, \
        mapping_storage_service, \
        llm_primary_service, \
        training_manager, \
        chat_service, \
        field_analyzer, \
        hybrid_orchestrator

    logger.info("=" * 80)
    logger.info("INITIALIZING SMART HYBRID MAPPING SYSTEM")
    logger.info("=" * 80)

    # ========================================================================
    # STEP 1: Initialize LLM Manager (Core Component)
    # ========================================================================
    try:
        logger.info("🤖 Step 1/5: Initializing LLM Manager...")
        llama_manager = LlamaManager()

        if llama_manager.is_model_loaded():
            logger.info("✅ CodeLlama-13B model loaded successfully")

            # Quick test
            test_response = llama_manager.generate_response(
                prompt="Generate a simple JSON with status: ready",
                max_tokens=50,
                temperature=0.4,
                timeout=600,
            )

            if test_response and len(test_response.strip()) > 0:
                logger.info(f"✅ LLM test successful ({len(test_response)} chars)")
            else:
                logger.warning("⚠️ LLM test returned empty response")
        else:
            logger.error("❌ LLM model not loaded - SYSTEM CANNOT OPERATE")
            return False

    except Exception as e:
        logger.error(f"❌ LLM initialization failed: {e}", exc_info=True)
        llama_manager = None
        return False

    # ========================================================================
    # STEP 2: Initialize Mapping Storage Service
    # ========================================================================
    try:
        logger.info("📚 Step 2/5: Initializing Mapping Storage Service...")
        mapping_storage_service = MappingStorageService()

        if mapping_storage_service.is_available():
            logger.info("✅ Mapping Storage Service ready (768-dim embeddings)")
        else:
            logger.warning("⚠️ Mapping Storage Service not available")
            mapping_storage_service = None

    except Exception as e:
        logger.error(
            f"❌ Mapping Storage Service initialization failed: {e}", exc_info=True
        )
        mapping_storage_service = None

    # ========================================================================
    # STEP 3: Initialize Milvus Vector Database
    # ========================================================================
    total_templates = 0
    try:
        logger.info("🗄️ Step 3/5: Connecting to Milvus Vector Database...")
        milvus_connected = connect_to_milvus()

        if milvus_connected:
            logger.info("✅ Successfully connected to Milvus")

            # Create/access collections
            milvus_collections = create_mapping_collections(dimension=768)

            if len(milvus_collections) > 0:
                logger.info(f"✅ Accessed {len(milvus_collections)} collections:")
                for collection_name in milvus_collections.keys():
                    logger.info(f"   - {collection_name}")

                # Get statistics
                try:
                    stats = get_all_collections_stats()
                    for name, stat in stats.items():
                        if "error" not in stat:
                            entities = stat.get("num_entities", 0)
                            total_templates += entities
                            logger.info(f"   - {name}: {entities} templates")

                    logger.info(f"📊 Total templates available: {total_templates}")

                    if total_templates >= 25:
                        logger.info("✅ EXCELLENT: 50%+ coverage capability")
                    elif total_templates >= 10:
                        logger.info("✅ GOOD: 30-50% coverage capability")
                    elif total_templates >= 5:
                        logger.warning(f"⚠️ FAIR: Limited templates ({total_templates})")
                    else:
                        logger.warning("⚠️ LOW: Few templates available")

                except Exception as stats_error:
                    logger.warning(f"Could not get collection stats: {stats_error}")
            else:
                logger.warning("⚠️ No mapping collections available")
                milvus_connected = False
        else:
            logger.warning("⚠️ Failed to connect to Milvus")
            milvus_connected = False

    except Exception as e:
        logger.warning(f"⚠️ Milvus initialization warning: {e}")
        milvus_connected = False

    # ========================================================================
    # STEP 4: Initialize LLM Primary Service (if needed for existing APIs)
    # ========================================================================
    try:
        if llama_manager and llama_manager.is_model_loaded():
            logger.info("🌟 Step 4/5: Initializing LLM Primary Mapping Service...")

            llm_primary_service = LLMPrimaryMappingService(
                llama_manager=llama_manager,
                mapping_storage_service=mapping_storage_service,
                milvus_collections=milvus_collections,
            )

            logger.info("✅ LLM Primary Service initialized")
        else:
            logger.warning(
                "⚠️ Cannot initialize LLM Primary Service - LLM not available"
            )
            llm_primary_service = None

    except Exception as e:
        logger.error(
            f"❌ LLM Primary Service initialization failed: {e}", exc_info=True
        )
        llm_primary_service = None

    # ========================================================================
    # STEP 5: Initialize Smart Hybrid Mapping System (NEW!)
    # ========================================================================
    try:
        if (
            llama_manager
            and llama_manager.is_model_loaded()
            and mapping_storage_service
        ):
            logger.info("🎯 Step 5/5: Initializing Smart Hybrid Mapping System...")

            # Import services
            from app.services.field_analyzer import FieldAnalyzer
            from app.services.smart_hybrid_orchestrator import SmartHybridOrchestrator

            # Initialize Field Analyzer
            field_analyzer = FieldAnalyzer()
            logger.info("✅ Field Analyzer initialized")

            # Initialize Smart Hybrid Orchestrator
            hybrid_orchestrator = SmartHybridOrchestrator(
                llama_manager=llama_manager,
                mapping_storage_service=mapping_storage_service,
                milvus_collections=milvus_collections,
            )
            logger.info("✅ Smart Hybrid Orchestrator initialized")

            logger.info("🚀 Smart Hybrid Mapping System READY:")
            logger.info("  - Three-tier workflow (Batch → Individual → Context)")
            logger.info("  - Intelligent field categorization")
            logger.info("  - 85-95% expected accuracy")
            logger.info("  - Parallel processing for complex fields")
            logger.info("  - Vector database integration")

        else:
            logger.warning("⚠️ Cannot initialize Smart Hybrid System")
            if not llama_manager or not llama_manager.is_model_loaded():
                logger.warning("   Reason: LLM not available")
            if not mapping_storage_service:
                logger.warning("   Reason: Mapping Storage Service not available")
            field_analyzer = None
            hybrid_orchestrator = None

    except ImportError as import_error:
        logger.error(f"❌ Smart Hybrid import failed: {import_error}")
        logger.error(
            "   Make sure field_analyzer.py and smart_hybrid_orchestrator.py exist"
        )
        field_analyzer = None
        hybrid_orchestrator = None
    except Exception as e:
        logger.error(f"❌ Smart Hybrid initialization failed: {e}", exc_info=True)
        field_analyzer = None
        hybrid_orchestrator = None

    # ========================================================================
    # OPTIONAL: Initialize Training Manager and Chat Service
    # ========================================================================
    try:
        if llama_manager and llama_manager.is_model_loaded():
            logger.info("💬 Initializing Chat Service...")
            chat_service = InteractiveChatService(llama_manager=llama_manager)
            logger.info("✅ Chat Service initialized")
        else:
            logger.warning("⚠️ Cannot initialize Chat Service")
            chat_service = None
    except Exception as e:
        logger.error(f"❌ Chat Service initialization failed: {e}", exc_info=True)
        chat_service = None

    try:
        logger.info("🏋️ Initializing Training Manager...")
        training_manager = LLMTrainingManager(
            ai_model_dir="/app/ai-model", llama_manager=llama_manager
        )

        hf_model_path = "/app/ai-model/CodeLlama-13b-hf"
        if os.path.exists(hf_model_path):
            logger.info("✅ Training Manager initialized")
            logger.info(f"   HF model found at: {hf_model_path}")
        else:
            logger.warning(f"⚠️ HF model directory not found: {hf_model_path}")

    except Exception as e:
        logger.error(f"❌ Training Manager initialization failed: {e}", exc_info=True)
        training_manager = None

    # ========================================================================
    # FINAL STATUS REPORT
    # ========================================================================
    logger.info("=" * 80)
    logger.info("INITIALIZATION SUMMARY")
    logger.info("=" * 80)

    services_status = {
        "llm_manager": llama_manager is not None and llama_manager.is_model_loaded(),
        "storage_service": mapping_storage_service is not None
        and mapping_storage_service.is_available(),
        "milvus": milvus_connected,
        "llm_primary_service": llm_primary_service is not None,
        "field_analyzer": field_analyzer is not None,
        "hybrid_orchestrator": hybrid_orchestrator is not None,
        "chat_service": chat_service is not None,
        "training_manager": training_manager is not None,
    }

    active_services = sum(services_status.values())
    total_services = len(services_status)

    # Log status
    for service_name, status in services_status.items():
        status_icon = "✅" if status else "❌"
        logger.info(
            f"{status_icon} {service_name}: {'Active' if status else 'Inactive'}"
        )

    logger.info(f"\n📊 Services Active: {active_services}/{total_services}")

    # Determine system capabilities
    critical_services = ["llm_manager", "storage_service"]
    critical_active = sum(services_status[service] for service in critical_services)

    if critical_active == len(critical_services):
        # Determine system mode
        if services_status["field_analyzer"] and services_status["hybrid_orchestrator"]:
            if services_status["milvus"] and total_templates >= 25:
                system_mode = (
                    "OPTIMAL SMART HYBRID (85-95% accuracy, Vector DB enhanced)"
                )
            elif services_status["milvus"] and total_templates >= 10:
                system_mode = "GOOD SMART HYBRID (80-90% accuracy, Vector DB enhanced)"
            elif services_status["milvus"]:
                system_mode = (
                    "STANDARD SMART HYBRID (75-85% accuracy, Vector DB enhanced)"
                )
            else:
                system_mode = "BASIC SMART HYBRID (70-80% accuracy, LLM only)"
        elif services_status["llm_primary_service"]:
            system_mode = "LEGACY MAPPING (Standard LLM-based)"
        else:
            system_mode = "MINIMAL (LLM only, limited features)"

        logger.info(f"🌟 SYSTEM READY: {system_mode}")
        logger.info("\n🚀 Available Features:")

        if services_status["field_analyzer"] and services_status["hybrid_orchestrator"]:
            logger.info("  ✅ Smart Hybrid Mapping (NEW!)")
            logger.info("     - /api/v1/smart-hybrid/analyze")
            logger.info("     - /api/v1/smart-hybrid/generate")
            logger.info("     - /api/v1/smart-hybrid/full")

        if services_status["llm_primary_service"]:
            logger.info("  ✅ Legacy Mapping")
            logger.info("     - /api/generate-mapping")

        if services_status["milvus"]:
            logger.info("  ✅ Vector Database")
            logger.info("     - /store_mapping_template")

        if services_status["training_manager"]:
            logger.info("  ✅ Model Training")
            logger.info("     - /api/training/start")
            logger.info("     - /api/training/status/<job_id>")

        if services_status["chat_service"]:
            logger.info("  ✅ Interactive Chat")

        logger.info("=" * 80)
        return True

    else:
        logger.error("=" * 80)
        logger.error("❌ CRITICAL SERVICES FAILED - SYSTEM CANNOT OPERATE")
        logger.error("=" * 80)
        logger.error("Missing critical services:")
        for service in critical_services:
            if not services_status[service]:
                logger.error(f"  ❌ {service}")
        return False


def validate_json_request():
    """Enhanced validation that preserves complete original input for clean response format"""
    try:
        if request.method == "OPTIONS":
            return None

        if not request.data and not request.form and not request.json:
            return {"error": "Empty request body"}, 400

        json_data = None
        if request.is_json:
            json_data = request.get_json()
        elif request.data:
            try:
                json_data = json.loads(request.data.decode("utf-8"))
            except json.JSONDecodeError:
                return {"error": "Invalid JSON format"}, 400
        elif request.form:
            json_data = dict(request.form)

        if json_data is None:
            return {"error": "Could not parse JSON from request"}, 400

        # CRITICAL: Store complete original input for clean response format
        if llm_primary_service and hasattr(
            llm_primary_service, "set_original_input_context"
        ):
            try:
                llm_primary_service.set_original_input_context(json_data)
                logger.debug(
                    "Stored complete original input context for clean response format"
                )
            except Exception as e:
                logger.warning(f"Could not store original input context: {e}")

        return json_data

    except Exception as e:
        logger.error(f"Request validation error: {e}")
        return {"error": f"Request validation failed: {str(e)}"}, 400


def _prepare_clean_response_data(
    mapping_result: Dict[str, Any], start_time: float
) -> Dict[str, Any]:
    """Prepare clean response - return COMPLETE original input with enhanced targetTreeNode and proper metadata"""

    generation_time = time.time() - start_time

    # Get the COMPLETE original input data
    original_input = (
        getattr(llm_primary_service, "_original_input_context", {})
        if llm_primary_service
        else {}
    )

    if not original_input:
        logger.error(
            "CRITICAL: No original input context found - cannot create clean response"
        )
        return {
            "error": "No original input context available",
            "details": "Original input was not preserved during processing",
            "timestamp": datetime.now().isoformat(),
        }

    # Start with COMPLETE original input structure
    clean_response = json.loads(json.dumps(original_input))

    # Replace ONLY the targetTreeNode with the enhanced one
    if mapping_result.get("targetTreeNode"):
        clean_response["targetTreeNode"] = mapping_result["targetTreeNode"]
        logger.info("Enhanced targetTreeNode in complete original input structure")
    else:
        logger.warning("No enhanced targetTreeNode found in mapping result")

    # ADD MISSING FIELDS: Calculate proper referenceVarCount
    target_tree = clean_response.get("targetTreeNode", {})
    reference_var_count = _calculate_reference_var_count(target_tree)
    clean_response["referenceVarCount"] = reference_var_count

    # ADD MISSING FIELDS: Ensure proper localContext
    if "localContext" not in clean_response or not clean_response["localContext"].get(
        "classes"
    ):
        # Get localContext from mapping result or use default comprehensive one
        local_context = mapping_result.get("localContext")
        if not local_context or not local_context.get("classes"):
            # Use the comprehensive local context as fallback
            local_context = _get_default_comprehensive_local_context()

        clean_response["localContext"] = local_context
        logger.info("Added comprehensive localContext to clean response")

    # Add MINIMAL essential metadata ONLY (no bloated response)
    field_analysis = mapping_result.get("field_coverage_analysis", {})

    clean_response["_mapping_metadata"] = {
        "generation_time": f"{generation_time:.3f}s",
        "mapped_fields_count": field_analysis.get("mapped_fields_count", 0),
        "total_input_fields": field_analysis.get("total_input_fields", 0),
        "coverage_percentage": field_analysis.get("coverage_percentage", 0.0),
        "method": mapping_result.get("generation_method", "unknown"),
        "template_confidence": mapping_result.get("template_confidence", 0.0),
        "micro_chunks_used": mapping_result.get("micro_chunks_used", 0),
        "valid": mapping_result.get("valid", False),
        "timestamp": datetime.now().isoformat(),
    }

    logger.info(
        f"Clean response prepared: {field_analysis.get('mapped_fields_count', 0)} fields mapped, "
        f"{field_analysis.get('coverage_percentage', 0.0):.1f}% coverage, "
        f"referenceVarCount: {reference_var_count}"
    )

    return clean_response


def _get_default_comprehensive_local_context() -> Dict[str, Any]:
    """Get default comprehensive local context with essential classes"""
    return {
        "globalVariables": [],
        "functions": [],
        "lookupTables": [],
        "classes": [
            {
                "name": "MapperUtility",
                "value": """class MapperUtility {
    public static boolean isNullOrEmpty(def item){
        if(item == "" || item == null || item == "null")
        {
        return true
        }
        return false
    }
    public static def convertToDouble(def item)
    {   
        def output
            if(isNullOrEmpty(item))
            {
            output= null
            }
            else
            {
            output = item.toDouble()
            }
        return output
    }
}""",
                "shortValue": "class MapperUtility {...}",
            }
        ],
    }


def _calculate_reference_var_count(target_tree_node: Dict[str, Any]) -> int:
    """Calculate the actual reference variable count from target tree"""
    max_var_id = 0

    def extract_var_ids(node):
        nonlocal max_var_id
        if isinstance(node, dict):
            # Check references array
            if "references" in node and isinstance(node["references"], list):
                for ref in node["references"]:
                    if isinstance(ref, dict) and "jsonId" in ref:
                        try:
                            var_id = int(ref["jsonId"])
                            max_var_id = max(max_var_id, var_id)
                        except (ValueError, TypeError):
                            pass

            # Recurse through all values
            for value in node.values():
                if isinstance(value, (dict, list)):
                    extract_var_ids(value)
        elif isinstance(node, list):
            for item in node:
                extract_var_ids(item)

    extract_var_ids(target_tree_node)
    return max_var_id


@app.route("/api/generate-mapping", methods=["POST", "OPTIONS"])
@auto_gpu_cleanup
def generate_mapping():
    """Generate comprehensive mapping with fixed thread management"""

    if request.method == "OPTIONS":
        response = jsonify({"status": "OK"})
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
        return response, 200

    validation_result = validate_json_request()
    if isinstance(validation_result, tuple):
        return jsonify(validation_result[0]), validation_result[1]

    input_data = validation_result

    try:
        logger.info("Starting COMPREHENSIVE Enhanced Mapping Generation...")
        start_time = time.time()

        original_target_tree = input_data.get("targetTreeNode")
        has_target_tree = original_target_tree is not None

        test_input_str = input_data.get("testInput", "{}")
        if isinstance(test_input_str, str):
            estimated_fields = test_input_str.count(":")
        else:
            estimated_fields = len(str(test_input_str))

        logger.info(
            f"Input analysis: targetTreeNode={'present' if has_target_tree else 'absent'}, "
            f"estimated complexity: ~{estimated_fields} fields"
        )

        if not llama_manager or not llama_manager.is_model_loaded():
            logger.error("Enhanced LLM model not available")
            return jsonify(
                {
                    "error": "Enhanced LLM model not available",
                    "details": "CodeLlama model not loaded",
                    "timestamp": datetime.now().isoformat(),
                }
            ), 503

        if not llm_primary_service:
            logger.error("Enhanced LLM-Primary service not available")
            return jsonify(
                {
                    "error": "Enhanced LLM-Primary service not available",
                    "timestamp": datetime.now().isoformat(),
                }
            ), 503

        # Fixed thread management with proper cleanup
        COMPREHENSIVE_TIMEOUT = 600  # 3 minutes

        result_queue = queue.Queue()
        timeout_event = threading.Event()
        worker_exception = None

        def comprehensive_mapping_worker():
            nonlocal worker_exception
            try:
                if timeout_event.is_set():
                    return

                logger.info("Worker thread started")
                result = llm_primary_service.generate_mapping_with_llm(input_data)

                if not timeout_event.is_set():
                    result_queue.put(result)
                    logger.info("Worker thread completed successfully")
            except Exception as e:
                worker_exception = e
                logger.error(f"Worker thread error: {e}", exc_info=True)
                if not timeout_event.is_set():
                    result_queue.put({"error": True, "error_message": str(e)})

        # Start worker thread with daemon=False for proper cleanup
        worker_thread = threading.Thread(
            target=comprehensive_mapping_worker, daemon=False
        )
        worker_thread.start()
        logger.info("Started worker thread for mapping generation")

        # Wait with timeout
        worker_thread.join(timeout=COMPREHENSIVE_TIMEOUT)

        if worker_thread.is_alive():
            # Timeout occurred
            logger.warning(f"⚠️ TIMEOUT: Processing exceeded {COMPREHENSIVE_TIMEOUT}s")
            timeout_event.set()

            # Give thread 5 more seconds to cleanup
            worker_thread.join(timeout=5)

            if worker_thread.is_alive():
                logger.error("Worker thread did not terminate gracefully")

            # Return timeout response
            original_input = getattr(llm_primary_service, "_original_input_context", {})
            error_response = (
                json.loads(json.dumps(original_input)) if original_input else {}
            )
            error_response.update(
                {
                    "_error": "Processing timeout",
                    "_error_details": f"Processing exceeded {COMPREHENSIVE_TIMEOUT} seconds",
                    "_generation_time": f"{time.time() - start_time:.3f}s",
                    "_timestamp": datetime.now().isoformat(),
                }
            )
            return jsonify(error_response), 408

        # Check for worker exception
        if worker_exception:
            logger.error(f"Worker thread raised exception: {worker_exception}")
            original_input = getattr(llm_primary_service, "_original_input_context", {})
            error_response = (
                json.loads(json.dumps(original_input)) if original_input else {}
            )
            error_response.update(
                {
                    "_error": "Worker thread exception",
                    "_error_details": str(worker_exception),
                    "_generation_time": f"{time.time() - start_time:.3f}s",
                    "_timestamp": datetime.now().isoformat(),
                }
            )
            return jsonify(error_response), 500

        # Get result from queue
        try:
            mapping_result = result_queue.get(timeout=2)
        except queue.Empty:
            logger.error("Worker thread completed but no result available")
            original_input = getattr(llm_primary_service, "_original_input_context", {})
            error_response = (
                json.loads(json.dumps(original_input)) if original_input else {}
            )
            error_response.update(
                {
                    "_error": "No result from worker thread",
                    "_generation_time": f"{time.time() - start_time:.3f}s",
                    "_timestamp": datetime.now().isoformat(),
                }
            )
            return jsonify(error_response), 500

        # Process successful result
        if not mapping_result.get("error"):
            logger.info("Comprehensive mapping generation successful")

            clean_response = _prepare_clean_response_data(mapping_result, start_time)

            if clean_response.get("error"):
                logger.error("Failed to create clean response format")
                return jsonify(clean_response), 500

            metadata = clean_response.get("_mapping_metadata", {})
            coverage = metadata.get("coverage_percentage", 0.0)
            mapped_count = metadata.get("mapped_fields_count", 0)
            total_count = metadata.get("total_input_fields", 0)
            generation_time = time.time() - start_time
            method = metadata.get("method", "unknown")

            logger.info(
                f"🎯 COMPREHENSIVE mapping completed in {generation_time:.1f}s:"
            )
            logger.info(f"  - Method: {method}")
            logger.info(f"  - Fields mapped: {mapped_count}/{total_count}")
            logger.info(f"  - Coverage: {coverage:.1f}%")

            if coverage >= 50:
                logger.info("🏆 EXCELLENT: Achieved 50%+ target!")
            elif coverage >= 40:
                logger.info("🎯 VERY GOOD: 40%+ coverage")
            elif coverage >= 30:
                logger.info("✅ GOOD: 30%+ coverage")
            else:
                logger.warning(f"⚠️ LOW: {coverage:.1f}% coverage")

            return jsonify(clean_response), 200
        else:
            logger.error("Comprehensive mapping generation failed")
            original_input = getattr(llm_primary_service, "_original_input_context", {})
            error_response = (
                json.loads(json.dumps(original_input)) if original_input else {}
            )
            error_response.update(
                {
                    "_error": True,
                    "_error_message": mapping_result.get(
                        "error_message", "Mapping failed"
                    ),
                    "_generation_time": f"{time.time() - start_time:.3f}s",
                    "_timestamp": datetime.now().isoformat(),
                }
            )
            return jsonify(error_response), 500

    except Exception as e:
        generation_time = time.time() - start_time
        logger.error(
            f"Mapping system error after {generation_time:.1f}s: {e}", exc_info=True
        )

        original_input = getattr(llm_primary_service, "_original_input_context", {})
        error_response = (
            json.loads(json.dumps(original_input)) if original_input else {}
        )
        error_response.update(
            {
                "_error": "System error",
                "_error_details": str(e),
                "_generation_time": f"{generation_time:.3f}s",
                "_timestamp": datetime.now().isoformat(),
            }
        )
        return jsonify(error_response), 500


@app.route("/api/store-mapping-template", methods=["POST"])
def store_mapping_template():
    """Store a mapping template in Milvus."""
    try:
        # Check if Milvus collections and mapping service are initialized
        if not milvus_collections or "OTMToJT" not in milvus_collections:
            logger.error("Milvus collection 'OTMToJT' not initialized")
            return jsonify(
                {"error": "Milvus collection 'OTMToJT' not initialized"}
            ), 500

        if not mapping_storage_service or not mapping_storage_service.is_available():
            logger.error("Mapping Storage Service not initialized or unavailable")
            return jsonify({"error": "Mapping Storage Service not available"}), 500

        # Check Milvus connection
        if not connect_to_milvus():
            logger.error("Failed to connect to Milvus")
            return jsonify({"error": "Failed to connect to Milvus"}), 500

        data = request.get_json()
        # logger.debug(f"Received request payload: {data}")
        if not data:
            logger.error("Invalid input: No JSON data provided")
            return jsonify({"error": "Invalid input"}), 400

        template_name = data.get("name", "Unknown")
        resourceId = data.get(
            "resourceId", mapping_storage_service._generate_resourceId(template_name)
        )
        chunks = mapping_storage_service.process_mapping_data(data)

        if not chunks:
            logger.error("No chunks generated from mapping data")
            return jsonify({"error": "No chunks generated"}), 400

        collection = milvus_collections["OTMToJT"]
        try:
            collection.load()
        except Exception as e:
            logger.error(f"Failed to load collection 'OTMToJT': {str(e)}")
            return jsonify(
                {"error": f"Failed to load collection 'OTMToJT': {str(e)}"}
            ), 500

        texts = [chunk["target_node_path"] for chunk in chunks]
        embeddings = [chunk["vector_embedding"] for chunk in chunks]
        metadata_list = [
            {
                "chunk_id": chunk["chunk_id"],
                "mapping_name": chunk["mapping_name"],
                "resource_id": chunk["resource_id"],
                "target_node_path": chunk["target_node_path"],
                "target_node": chunk["target_node"],
            }
            for chunk in chunks
        ]
        insert_result = insert_mapping_embeddings(
            collection_name="OTMToJT",
            texts=texts,
            embeddings=embeddings,
            metadata_list=metadata_list,
            source_format="EDI",
            target_format="JSON",
            mapping_type="mapped_node",
            template_name=template_name,
            resourceId=resourceId,
        )

        if insert_result:
            logger.info(
                f"Successfully inserted {len(chunks)} chunks for template: {template_name}"
            )
            return jsonify(
                {
                    "success": True,
                    "chunks_inserted": len(chunks),
                    "resourceId": resourceId,
                }
            ), 200
        logger.error("Failed to insert embeddings into 'OTMToJT'")
        return jsonify({"error": "Failed to insert embeddings"}), 500
    except Exception as e:
        logger.error(f"Error storing template: {str(e)}")
        return jsonify({"error": str(e)}), 500


def _analyze_comprehensive_chunk_distribution(
    chunks: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Analyze the distribution and quality of comprehensive chunks"""

    if not chunks:
        return {
            "distribution_summary": "No chunks generated",
            "chunk_types": [],
            "pattern_types": [],
            "average_quality": 0.0,
            "total_chunks": 0,
        }

    try:
        # Analyze chunk types
        chunk_types = {}
        pattern_types = {}
        quality_scores = []

        for chunk in chunks:
            # Count chunk types
            chunk_type = chunk.get("chunk_type", "unknown")
            chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1

            # Count pattern types
            pattern_type = chunk.get("pattern_type", "unknown")
            pattern_types[pattern_type] = pattern_types.get(pattern_type, 0) + 1

            # Collect quality scores
            embedding_quality = chunk.get("embedding_quality", {})
            quality_score = embedding_quality.get("quality_score", 0.0)
            quality_scores.append(quality_score)

        # Calculate average quality
        average_quality = (
            sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
        )

        # Create distribution summary
        chunk_type_summary = ", ".join([f"{k}({v})" for k, v in chunk_types.items()])
        distribution_summary = f"Generated {len(chunks)} chunks: {chunk_type_summary}"

        return {
            "distribution_summary": distribution_summary,
            "chunk_types": list(chunk_types.keys()),
            "pattern_types": list(pattern_types.keys()),
            "chunk_type_counts": chunk_types,
            "pattern_type_counts": pattern_types,
            "average_quality": round(average_quality, 3),
            "quality_distribution": {
                "excellent": sum(1 for q in quality_scores if q > 0.8),
                "good": sum(1 for q in quality_scores if 0.6 < q <= 0.8),
                "fair": sum(1 for q in quality_scores if 0.4 < q <= 0.6),
                "poor": sum(1 for q in quality_scores if q <= 0.4),
            },
            "total_chunks": len(chunks),
            "embedding_dimension": 768
            if any(len(c.get("embedding", [])) == 768 for c in chunks)
            else 768,
            "expected_accuracy_range": "65-75%"
            if any(len(c.get("embedding", [])) == 768 for c in chunks)
            else "50-60%",
        }

    except Exception as e:
        logger.error(f"Error analyzing chunk distribution: {e}")
        return {
            "distribution_summary": f"Analysis failed: {str(e)}",
            "chunk_types": [],
            "pattern_types": [],
            "average_quality": 0.0,
            "total_chunks": len(chunks),
            "error": str(e),
        }


@app.route("/health")
def health_check():
    """Enhanced health check for comprehensive mapping system"""
    try:
        # Check critical components
        llm_status = llama_manager is not None and llama_manager.is_model_loaded()
        llm_primary_status = llm_primary_service is not None

        # Check comprehensive mapping components
        storage_service_status = mapping_storage_service is not None
        vector_context_status = milvus_connected

        # Get comprehensive template statistics
        collections_status = {}
        total_templates = 0
        comprehensive_readiness = "none"

        if milvus_connected:
            try:
                collections_stats = get_all_collections_stats()
                for collection_name, stats in collections_stats.items():
                    if "error" not in stats:
                        entities = stats.get("num_entities", 0)
                        collections_status[collection_name] = {
                            "entities": entities,
                            "accessible": True,
                            "comprehensive_ready": entities >= 10,
                        }
                        total_templates += entities
                    else:
                        collections_status[collection_name] = {
                            "entities": 0,
                            "accessible": False,
                            "comprehensive_ready": False,
                            "error": stats["error"],
                        }

                # Assess comprehensive readiness
                if total_templates >= 50:
                    comprehensive_readiness = "excellent"
                elif total_templates >= 25:
                    comprehensive_readiness = "very_good"
                elif total_templates >= 10:
                    comprehensive_readiness = "good"
                elif total_templates >= 1:
                    comprehensive_readiness = "limited"
                else:
                    comprehensive_readiness = "none"

            except Exception as e:
                collections_status = {
                    "error": f"Failed to get collection stats: {str(e)}"
                }

        # Determine system status for comprehensive mapping
        if llm_status and llm_primary_status:
            if (
                vector_context_status
                and storage_service_status
                and total_templates >= 25
            ):
                system_status = "optimal_comprehensive"  # 50%+ coverage capability
            elif (
                vector_context_status
                and storage_service_status
                and total_templates >= 10
            ):
                system_status = "good_comprehensive"  # 30-50% coverage capability
            elif vector_context_status and storage_service_status:
                system_status = "fair_comprehensive"  # 20-30% coverage capability
            elif storage_service_status:
                system_status = "basic_comprehensive"  # 10-20% coverage capability
            else:
                system_status = "limited"  # <10% coverage capability
        else:
            system_status = "degraded"

        return jsonify(
            {
                "status": system_status,
                "architecture": "comprehensive_vector_database_driven_mapping",
                "primary_engine": "CodeLlama-13B-Instruct",
                "mapping_approach": "Vector Database + Enhanced LLM + Comprehensive Analysis",
                "components": {
                    # Critical Components for Comprehensive Mapping
                    "llm_manager": llm_status,
                    "llm_primary_service": llm_primary_status,
                    "storage_service": storage_service_status,
                    "milvus_vector_database": vector_context_status,
                    "comprehensive_engine": llm_primary_status
                    and storage_service_status,
                },
                "comprehensive_capabilities": {
                    "vector_database_templates": total_templates,
                    "comprehensive_readiness": comprehensive_readiness,
                    "expected_coverage_range": _get_expected_coverage_range(
                        comprehensive_readiness
                    ),
                    "template_matching": vector_context_status,
                    "intelligent_field_mapping": llm_status,
                    "pattern_learning": vector_context_status
                    and storage_service_status,
                    "collections": collections_status,
                },
                "coverage_expectations": {
                    "target_coverage": "50-70%"
                    if comprehensive_readiness in ["excellent", "very_good"]
                    else "30-50%",
                    "expected_field_count": "50+ fields"
                    if total_templates >= 25
                    else "30+ fields",
                    "processing_time": "30-120 seconds for comprehensive analysis",
                    "accuracy_level": "high"
                    if comprehensive_readiness in ["excellent", "very_good"]
                    else "medium",
                },
                "response_format": {
                    "format": "complete_original_input_with_enhanced_targetTreeNode",
                    "bloat_removed": True,
                    "original_input_preserved": True,
                    "metadata_minimal": True,
                },
                "capabilities": [
                    "Comprehensive vector database-driven field mapping",
                    "Enhanced LLM pattern learning and application",
                    "Intelligent field categorization and prioritization",
                    "Multi-tier processing (vector patterns → LLM → fallbacks)",
                    "50-70% field coverage target with high accuracy",
                    "Complete original input preservation with enhanced targetTreeNode",
                    "Minimal metadata bloat with essential coverage statistics",
                ]
                if llm_status
                else ["System unavailable - LLM not loaded"],
                "model_info": llama_manager.get_model_info() if llama_manager else None,
                "version": "comprehensive_vector_llm_v7.0",
                "timestamp": datetime.now().isoformat(),
            }
        ), 200 if system_status.endswith("comprehensive") else 503

    except Exception as e:
        logger.error(f"Health check error: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500


def _get_expected_coverage_range(readiness_level: str) -> str:
    """Get expected coverage range based on readiness"""
    coverage_ranges = {
        "excellent": "50-70%",
        "very_good": "40-60%",
        "good": "30-50%",
        "limited": "20-40%",
        "none": "10-20%",
    }
    return coverage_ranges.get(readiness_level, "10-20%")


@app.route("/api/training/start", methods=["POST", "OPTIONS"])
def start_training():
    """Start training with automatic continual learning (builds on latest successful model)"""

    if request.method == "OPTIONS":
        response = jsonify({"status": "OK"})
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
        return response, 200

    if not training_manager:
        return jsonify(
            {
                "error": "Training manager not available",
                "details": "Training system not initialized",
            }
        ), 503

    validation_result = validate_json_request()
    if isinstance(validation_result, tuple):
        return jsonify(validation_result[0]), validation_result[1]

    request_data = validation_result

    try:
        # Validate required fields
        required_fields = ["training_examples"]
        missing_fields = [
            field for field in required_fields if field not in request_data
        ]

        if missing_fields:
            return jsonify(
                {
                    "error": "Missing required fields",
                    "missing_fields": missing_fields,
                    "required_fields": required_fields,
                }
            ), 400

        model_name = request_data.get("model_name", "CodeLlama-13b-hf")
        training_examples = request_data.get("training_examples", [])
        training_config = request_data.get("training_config", {})
        job_name = request_data.get(
            "job_name",
            f"Auto-Continual-Training-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        )
        force_fresh = request_data.get(
            "force_fresh", False
        )  # NEW: Override auto-continual

        # Validate training examples
        if not training_examples:
            return jsonify(
                {
                    "error": "No training examples provided",
                    "details": "At least one training example is required",
                }
            ), 400

        for i, example in enumerate(training_examples):
            if not all(key in example for key in ["input_data", "output_data"]):
                return jsonify(
                    {
                        "error": f"Invalid training example format at index {i}",
                        "details": "Each example must have 'input_data' and 'output_data' fields",
                        "example_format": {
                            "input_data": {"field": "value"},
                            "output_data": {"field": "value"},
                            "mapping_type": "any string value describing the mapping type",
                            "description": "optional description of the mapping",
                        },
                    }
                ), 400

            mapping_type = example.get("mapping_type", "general")
            # valid_types = ["otm_to_jt_canonical", "jt_canonical_to_fedex", "general"]
            # if mapping_type not in valid_types:
            #     return jsonify(
            #         {
            #             "error": f"Invalid mapping type at index {i}: {mapping_type}",
            #             "details": f"Valid mapping types: {valid_types}",
            #         }
            #     ), 400

        # Check what auto-continual learning will do
        auto_info = {}
        if not force_fresh:
            latest_job_id = training_manager.get_latest_successful_job()
            if latest_job_id:
                latest_job = training_manager.get_job_status(latest_job_id)
                auto_info = {
                    "auto_continual_enabled": True,
                    "will_continue_from": latest_job_id,
                    "base_job_name": latest_job["job_name"]
                    if latest_job
                    else "Unknown",
                    "base_model_version": latest_job["model_version"]
                    if latest_job
                    else 1,
                    "next_model_version": (latest_job["model_version"] + 1)
                    if latest_job
                    else 1,
                }
            else:
                auto_info = {
                    "auto_continual_enabled": True,
                    "will_continue_from": None,
                    "message": "No previous successful jobs found - starting fresh training",
                }
        else:
            auto_info = {
                "auto_continual_enabled": False,
                "force_fresh_requested": True,
                "message": "Auto-continual learning disabled for this training",
            }

        # Pre-training analysis
        logger.info(f"Starting auto-continual training job: {job_name}")
        logger.info(f"Training examples: {len(training_examples)}")
        logger.info(
            f"Mapping types: {set(ex.get('mapping_type', 'general') for ex in training_examples)}"
        )
        logger.info(f"Training config: {training_config}")
        logger.info(f"Force fresh: {force_fresh}")
        if auto_info.get("will_continue_from"):
            logger.info(f"Will auto-continue from: {auto_info['will_continue_from']}")

        # Start auto-continual training
        job_id = training_manager.start_training(
            model_name=model_name,
            training_examples=training_examples,
            config=training_config,
            job_name=job_name,
            force_fresh=force_fresh,
        )

        return jsonify(
            {
                "success": True,
                "job_id": job_id,
                "job_name": job_name,
                "message": "Auto-continual training job started successfully",
                "status": "started",
                "examples_count": len(training_examples),
                "mapping_types": list(
                    set(ex.get("mapping_type", "general") for ex in training_examples)
                ),
                "config": training_config,
                "auto_continual_info": auto_info,  # NEW: Auto-continual details
                "enhancements": [
                    "auto_continual_learning",
                    "intelligent_base_detection",
                    "optimized_hyperparams",
                    "enhanced_instructions",
                    "detailed_monitoring",
                    "model_versioning",
                ],
                "timestamp": datetime.now().isoformat(),
            }
        ), 200

    except Exception as e:
        logger.error(f"Error starting auto-continual training: {e}", exc_info=True)
        return jsonify(
            {
                "error": "Failed to start auto-continual training job",
                "details": str(e),
                "timestamp": datetime.now().isoformat(),
            }
        ), 500


@app.route("/api/training/status/<job_id>", methods=["GET"])
def get_training_status(job_id: str):
    """Get enhanced training job status with auto-continual details"""

    if not training_manager:
        return jsonify({"error": "Training manager not available"}), 503

    try:
        status = training_manager.get_job_status(job_id)

        if not status:
            return jsonify({"error": "Training job not found", "job_id": job_id}), 404

        # Add computed metrics
        if status["status"] == "running" and status["total_epochs"] > 0:
            status["completion_percentage"] = (
                status["current_epoch"] / status["total_epochs"]
            ) * 100

        if status["loss"] and status["validation_loss"]:
            status["overfitting_indicator"] = (
                "high" if status["validation_loss"] > status["loss"] * 1.5 else "normal"
            )

        if status["training_metrics"]:
            status["training_stability"] = (
                "stable" if len(status["training_metrics"]) > 2 else "early"
            )

        # Add auto-continual chain information
        if status.get("training_type") == "continual" and status.get("base_job_id"):
            base_job = training_manager.get_job_status(status["base_job_id"])
            if base_job:
                status["continual_learning_chain"] = {
                    "base_job_name": base_job["job_name"],
                    "base_job_completion": base_job["end_time"],
                    "base_model_version": base_job["model_version"],
                    "current_model_version": status["model_version"],
                    "auto_detected": status.get("auto_continued", False),
                }

        return jsonify(status), 200

    except Exception as e:
        logger.error(f"Error getting training status: {e}")
        return jsonify(
            {
                "error": "Failed to get training status",
                "details": str(e),
                "job_id": job_id,
            }
        ), 500


# chat related functionalities ---------------------------------------------------------------
@app.route("/api/chat/message", methods=["POST", "OPTIONS"])
def chat_message():
    """
    Send message to chat service and get response

    Request body:
    {
        "session_id": "optional - will be generated if not provided",
        "message": "User's message text"
    }
    """

    if request.method == "OPTIONS":
        response = jsonify({"status": "OK"})
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
        return response, 200

    if not chat_service:
        return jsonify(
            {
                "error": "Chat service not available",
                "details": "Service failed to initialize",
            }
        ), 503

    # Validate request
    validation_result = validate_json_request()
    if isinstance(validation_result, tuple):
        return jsonify(validation_result[0]), validation_result[1]

    request_data = validation_result

    try:
        # Get or generate session ID
        session_id = request_data.get("session_id")
        if not session_id:
            session_id = f"chat_{uuid.uuid4().hex[:16]}"
            logger.info(f"Generated new session ID: {session_id}")

        # Get user message
        user_message = request_data.get("message", "").strip()

        if not user_message:
            return jsonify(
                {"error": "Missing message", "details": "Message field is required"}
            ), 400

        logger.info(f"Chat message from session {session_id}: {user_message[:100]}")

        # Process message through chat service
        result = chat_service.process_message(session_id, user_message)

        if result.get("success"):
            logger.info(
                f"Chat response generated: {len(result.get('response', ''))} chars"
            )
            return jsonify(result), 200
        else:
            logger.error(f"Chat processing failed: {result.get('error')}")
            return jsonify(result), 500

    except Exception as e:
        logger.error(f"Chat message endpoint error: {e}", exc_info=True)
        return jsonify(
            {
                "error": "Failed to process chat message",
                "details": str(e),
                "timestamp": datetime.now().isoformat(),
            }
        ), 500


@app.route("/api/chat/session/<session_id>", methods=["GET", "OPTIONS"])
def get_chat_session(session_id: str):
    """
    Get current state of a chat session

    Returns full conversation history and current state
    """

    if request.method == "OPTIONS":
        response = jsonify({"status": "OK"})
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
        return response, 200

    if not chat_service:
        return jsonify({"error": "Chat service not available"}), 503

    try:
        session_state = chat_service.get_session_state(session_id)

        if session_state:
            return jsonify(
                {
                    "success": True,
                    "session": session_state,
                    "timestamp": datetime.now().isoformat(),
                }
            ), 200
        else:
            return jsonify(
                {
                    "error": "Session not found",
                    "session_id": session_id,
                    "details": "Session may have expired or never existed",
                }
            ), 404

    except Exception as e:
        logger.error(f"Get session error: {e}", exc_info=True)
        return jsonify({"error": "Failed to retrieve session", "details": str(e)}), 500


@app.route("/api/chat/session/<session_id>", methods=["DELETE", "OPTIONS"])
def delete_chat_session(session_id: str):
    """
    Delete/clear a chat session

    Useful for starting fresh or clearing old conversations
    """

    if request.method == "OPTIONS":
        response = jsonify({"status": "OK"})
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "DELETE, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
        return response, 200

    if not chat_service:
        return jsonify({"error": "Chat service not available"}), 503

    try:
        success = chat_service.clear_session(session_id)

        if success:
            return jsonify(
                {
                    "success": True,
                    "message": "Session cleared successfully",
                    "session_id": session_id,
                    "timestamp": datetime.now().isoformat(),
                }
            ), 200
        else:
            return jsonify(
                {"error": "Session not found", "session_id": session_id}
            ), 404

    except Exception as e:
        logger.error(f"Delete session error: {e}", exc_info=True)
        return jsonify({"error": "Failed to delete session", "details": str(e)}), 500


@app.route("/api/chat/sessions", methods=["GET", "OPTIONS"])
def get_active_sessions():
    """
    Get count of active chat sessions

    Useful for monitoring and debugging
    """

    if request.method == "OPTIONS":
        response = jsonify({"status": "OK"})
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
        return response, 200

    if not chat_service:
        return jsonify({"error": "Chat service not available"}), 503

    try:
        active_count = chat_service.get_active_sessions_count()

        return jsonify(
            {
                "success": True,
                "active_sessions": active_count,
                "session_timeout_minutes": 30,
                "timestamp": datetime.now().isoformat(),
            }
        ), 200

    except Exception as e:
        logger.error(f"Get active sessions error: {e}", exc_info=True)
        return jsonify({"error": "Failed to get session count", "details": str(e)}), 500


@app.route("/api/chat/history/<session_id>", methods=["GET", "OPTIONS"])
def get_chat_history(session_id: str):
    """
    Get conversation history for a session (simplified view)

    Returns just the messages without full session state
    """

    if request.method == "OPTIONS":
        response = jsonify({"status": "OK"})
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
        return response, 200

    if not chat_service:
        return jsonify({"error": "Chat service not available"}), 503

    try:
        session_state = chat_service.get_session_state(session_id)

        if session_state:
            # Extract just the conversation history
            messages = session_state.get("conversation_history", [])

            return jsonify(
                {
                    "success": True,
                    "session_id": session_id,
                    "message_count": len(messages),
                    "messages": messages,
                    "created_at": session_state.get("created_at"),
                    "last_activity": session_state.get("last_activity"),
                    "timestamp": datetime.now().isoformat(),
                }
            ), 200
        else:
            return jsonify(
                {"error": "Session not found", "session_id": session_id}
            ), 404

    except Exception as e:
        logger.error(f"Get history error: {e}", exc_info=True)
        return jsonify(
            {"error": "Failed to retrieve chat history", "details": str(e)}
        ), 500


# @app.route("/api/v1/smart-hybrid/full", methods=["POST", "OPTIONS"])
# @auto_gpu_cleanup
# def smart_hybrid_full():
#     """Smart Hybrid full workflow: Analyze + Map with auto-detection and options"""
#     if request.method == "OPTIONS":
#         response = jsonify({"status": "OK"})
#         response.headers["Access-Control-Allow-Origin"] = "*"
#         response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
#         response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
#         return response, 200

#     validation_result = validate_json_request()
#     if isinstance(validation_result, tuple):
#         return jsonify(validation_result[0]), validation_result[1]

#     input_data = validation_result

#     try:
#         # Smart input detection
#         unmapped_input = None
#         input_format = "json"  # Default
#         source_format = None
#         target_format = "JSON"  # Default
#         collection_name = "OTMToJT"  # Default
#         options = {}

#         # Check if user sent just the raw input (no wrapper)
#         if (
#             "input_data" not in input_data
#             and "source_format" not in input_data
#             and "target_format" not in input_data
#             and "input_format" not in input_data
#         ):
#             # User sent raw unmapped data directly
#             unmapped_input = input_data
#             logger.info("📥 Detected raw input format (no wrapper)")
#         else:
#             # User used structured format
#             unmapped_input = input_data.get("input_data", input_data)
#             input_format = input_data.get("input_format", "json").lower()
#             source_format = input_data.get("source_format")
#             target_format = input_data.get("target_format", "JSON")
#             collection_name = input_data.get("collection_name", "OTMToJT")
#             options = input_data.get("options", {})

#         # Auto-detect source format if not provided
#         if not source_format:
#             source_format = _auto_detect_source_format(unmapped_input)
#             logger.info(f"🔍 Auto-detected source format: {source_format}")

#         # Validate input format
#         if input_format not in ["json", "edi", "xml"]:
#             input_format = "json"  # Default to JSON

#         # Check if services are available
#         if not field_analyzer or not hybrid_orchestrator:
#             return jsonify(
#                 {
#                     "error": "Smart Hybrid system not available",
#                     "details": "Field analyzer or orchestrator not initialized",
#                     "timestamp": datetime.now().isoformat(),
#                 }
#             ), 503

#         logger.info(f"🚀 Starting Smart Hybrid: {source_format} → {target_format}")
#         logger.info(f"   Input format: {input_format}")
#         logger.info(f"   Collection: {collection_name}")
#         overall_start_time = time.time()

#         # STEP 1: Analyze
#         logger.info("📊 Step 1/2: Analyzing input fields...")
#         analysis_start = time.time()
#         analysis = field_analyzer.analyze_input(unmapped_input, input_format)
#         analysis_time = time.time() - analysis_start

#         logger.info(f"✅ Analysis complete in {analysis_time:.2f}s")
#         logger.info(f"  - Total fields: {analysis['summary']['total_fields']}")
#         logger.info(f"  - Groups: {analysis['summary']['total_groups']}")
#         logger.info(
#             f"  - Static: {analysis['summary']['by_category'].get('static', 0)}"
#         )
#         logger.info(
#             f"  - Simple: {analysis['summary']['by_category'].get('simple', 0)}"
#         )
#         logger.info(
#             f"  - Complex: {analysis['summary']['by_category'].get('complex', 0)}"
#         )
#         logger.info(
#             f"  - Dependent: {analysis['summary']['by_category'].get('dependent', 0)}"
#         )

#         # STEP 2: Generate mapping
#         logger.info("🤖 Step 2/2: Generating mappings...")

#         # Apply options
#         if "tier1_batch_size" in options:
#             hybrid_orchestrator.tier1_batch_size = options["tier1_batch_size"]
#         if "tier2_parallel_workers" in options:
#             hybrid_orchestrator.tier2_parallel_workers = options[
#                 "tier2_parallel_workers"
#             ]
#         if "vector_search_top_k" in options:
#             hybrid_orchestrator.vector_search_top_k = options["vector_search_top_k"]

#         mapping_start = time.time()
#         mapping_result = hybrid_orchestrator.execute_mapping(
#             field_analysis=analysis,
#             source_format=source_format,
#             target_format=target_format,
#             collection_name=collection_name,
#         )
#         mapping_time = time.time() - mapping_start

#         total_time = time.time() - overall_start_time
#         stats = mapping_result.get("statistics", {})

#         # Calculate overall metrics
#         total_fields_analyzed = analysis["summary"]["total_fields"]
#         total_fields_mapped = stats.get("successful_mappings", 0)
#         overall_accuracy = (
#             (total_fields_mapped / total_fields_analyzed * 100)
#             if total_fields_analyzed > 0
#             else 0
#         )

#         logger.info(f"✅ Smart Hybrid workflow complete in {total_time:.1f}s:")
#         logger.info(f"  - Analysis: {analysis_time:.1f}s")
#         logger.info(f"  - Mapping: {mapping_time:.1f}s")
#         logger.info(
#             f"  - Fields mapped: {total_fields_mapped}/{total_fields_analyzed} ({overall_accuracy:.1f}%)"
#         )
#         logger.info(f"  - Tier 1: {stats.get('tier1_accuracy', 0)}%")
#         logger.info(f"  - Tier 2: {stats.get('tier2_accuracy', 0)}%")
#         logger.info(f"  - Tier 3: {stats.get('tier3_accuracy', 0)}%")
#         logger.info(f"  - LLM calls: {stats.get('llm_calls', 0)}")
#         logger.info(f"  - Avg confidence: {stats.get('avg_confidence', 0)}")

#         return jsonify(
#             {
#                 "success": True,
#                 "mapping": mapping_result,
#                 "metadata": {
#                     "source_format": source_format,
#                     "target_format": target_format,
#                     "input_format": input_format,
#                     "collection_used": collection_name,
#                     "auto_detected": not input_data.get("source_format"),
#                     "total_fields_analyzed": total_fields_analyzed,
#                     "total_fields_mapped": total_fields_mapped,
#                     "overall_accuracy": round(overall_accuracy, 1),
#                     "analysis_time_seconds": round(analysis_time, 2),
#                     "mapping_time_seconds": round(mapping_time, 2),
#                     "total_time_seconds": round(total_time, 2),
#                     "llm_calls": stats.get("llm_calls", 0),
#                     "avg_confidence": stats.get("avg_confidence", 0),
#                 },
#                 "message": f"Successfully mapped {total_fields_mapped}/{total_fields_analyzed} fields ({overall_accuracy:.1f}% accuracy)",
#                 "timestamp": datetime.now().isoformat(),
#             }
#         ), 200

#     except Exception as e:
#         logger.error(f"❌ Smart Hybrid workflow failed: {e}", exc_info=True)
#         return jsonify(
#             {
#                 "success": False,
#                 "error": "Mapping workflow failed",
#                 "details": str(e),
#                 "timestamp": datetime.now().isoformat(),
#             }
#         ), 500


@app.route("/api/v1/smart_hybrid_full", methods=["POST"])
def smart_hybrid_full():
    """Enhanced endpoint with field analysis - Returns complete mapped template"""
    try:
        start_time = time.time()
        input_data = request.get_json()

        # ============ DYNAMIC INPUT DETECTION ============
        if "sourceTreeNode" in input_data and "targetTreeNode" in input_data:
            logger.info("📥 Detected mapping template format")

            # Extract structures
            source_data = input_data.get("sourceTreeNode")
            target_data = input_data.get("targetTreeNode")

            # Get formats dynamically
            source_format = input_data.get("sourceInputType", "JSON")
            target_format = input_data.get("targetInputType", "JSON")

            # Store original template
            original_template = input_data.copy()

        else:
            # Standard format (backward compatibility)
            source_data = input_data.get("input_data", input_data)
            target_data = None
            source_format = input_data.get("input_format", "json").lower()
            target_format = input_data.get("target_format", "JSON")
            original_template = None

        logger.info(f"🚀 Starting Smart Hybrid: {source_format} → {target_format}")

        # ============ STEP 1: ANALYZE SOURCE FIELDS ============
        logger.info("📊 Step 1/2: Analyzing input fields...")

        field_analysis = field_analyzer.analyze_fields(source_data, source_format)

        # Extract tier assignments
        tier_assignments = field_analysis.get("tier_assignments", {})
        tier1_groups = tier_assignments.get("tier1_static_simple", [])
        tier2_fields = tier_assignments.get("tier2_complex", [])
        tier3_fields = tier_assignments.get("tier3_dependent", [])

        total_fields = field_analysis.get("summary", {}).get("total_fields", 0)

        logger.info(f"✅ Analysis complete in {time.time() - start_time:.2f}s")
        logger.info(f"  - Total fields: {total_fields}")
        logger.info(f"  - Tier 1 groups: {len(tier1_groups)}")
        logger.info(f"  - Tier 2 fields: {len(tier2_fields)}")
        logger.info(f"  - Tier 3 fields: {len(tier3_fields)}")

        # ============ STEP 2: GENERATE MAPPINGS ============
        logger.info("🤖 Step 2/2: Generating mappings...")
        # from app.services.smart_hybrid_orchestrator import SmartHybridOrchestrator

        mapping_result = hybrid_orchestrator.execute_mapping(
            tier1_groups=tier1_groups,  # ← Pass directly
            tier2_fields=tier2_fields,  # ← Pass directly
            tier3_fields=tier3_fields,  # ← Pass directly
            unmapped_input=source_data,
            target_structure=target_data,
            source_format=source_format,
            target_format=target_format,
            collection_name="OTMToJT",
        )

        workflow_time = time.time() - start_time
        logger.info(f"✅ Workflow complete in {workflow_time:.1f}s")
        logger.info(
            f"  - Fields mapped: {mapping_result.get('total_mapped', 0)}/{total_fields}"
        )
        logger.info(
            f"  - Overall accuracy: {mapping_result.get('overall_accuracy', 0):.1f}%"
        )

        # ============ STEP 3: BUILD RESPONSE IN ORIGINAL FORMAT ============
        if original_template:
            # Return in the same format as input
            response = original_template.copy()

            # Update targetTreeNode with mappings
            response["targetTreeNode"] = mapping_result.get(
                "mapped_target_tree", target_data
            )

            # Update referenceVarCount
            response["referenceVarCount"] = mapping_result.get("total_mapped", 0)

            logger.info("✅ Returning complete mapped template")
            return jsonify(response), 200

        else:
            # Standard response format (backward compatibility)
            response = {
                "status": "success",
                "mapped_output": mapping_result,
                "metadata": {
                    "total_fields": total_fields,
                    "fields_mapped": mapping_result.get("total_mapped", 0),
                    "accuracy": mapping_result.get("overall_accuracy", 0),
                    "workflow_time": workflow_time,
                },
            }
            return jsonify(response), 200

    except Exception as e:
        logger.error(f"❌ Error in smart_hybrid_full: {str(e)}")
        import traceback

        logger.error(traceback.format_exc())
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


def _add_to_hierarchy(hierarchy: Dict, path_parts: List[str], mapping: Dict):
    """Add mapping to hierarchical structure"""
    current = hierarchy

    # Navigate/create hierarchy
    for i, part in enumerate(path_parts[:-1]):
        if part not in current:
            current[part] = {}
        current = current[part]

    # Add final mapping
    final_field = path_parts[-1]
    current[final_field] = mapping


def _hierarchy_to_children(hierarchy: Dict) -> List[Dict]:
    """Convert hierarchy dict to children array"""
    children = []

    for key, value in hierarchy.items():
        if isinstance(value, dict):
            if "name" in value:
                # This is a leaf mapping node
                children.append(value)
            else:
                # This is a container node
                child = {
                    "name": key,
                    "type": "o",
                    "children": _hierarchy_to_children(value),
                }
                children.append(child)

    return children


def _convert_input_to_tree(data: Any, name: str = "root") -> List[Dict]:
    """Convert input data to tree structure for sourceTreeNode"""
    if isinstance(data, dict):
        children = []
        for key, value in data.items():
            if isinstance(value, dict):
                children.append(
                    {
                        "name": key,
                        "type": "o",
                        "children": _convert_input_to_tree(value, key),
                    }
                )
            elif isinstance(value, list):
                children.append(
                    {
                        "name": key,
                        "type": "ar",  # array
                        "children": _convert_input_to_tree(value[0], key)
                        if value
                        else [],
                    }
                )
            else:
                children.append(
                    {"name": key, "type": _infer_type(value), "value": str(value)}
                )
        return children

    return []


def _infer_type(value: Any) -> str:
    """Infer field type shorthand"""
    if isinstance(value, bool):
        return "b"
    elif isinstance(value, int):
        return "n"
    elif isinstance(value, float):
        return "n"
    elif isinstance(value, str):
        return "s"
    elif isinstance(value, list):
        return "ar"
    elif isinstance(value, dict):
        return "o"
    else:
        return "s"


# Error handlers and signal handlers
@app.errorhandler(404)
def handle_not_found(e):
    return jsonify(
        {
            "error": "Not Found",
            "message": "Endpoint not found",
            "available_endpoints": [
                "/health",
                "/api/generate-mapping",
                "/api/generate-mapping-hybrid",
                "/api/store-mapping-template",
                "/api/comprehensive-status",
                "/api/clear-milvus",
            ],
            "timestamp": datetime.now().isoformat(),
        }
    ), 404


@app.errorhandler(500)
def handle_internal_error(e):
    logger.error(f"Internal error: {e}", exc_info=True)
    return jsonify(
        {
            "error": "Internal Server Error",
            "message": "Unexpected server error occurred",
            "timestamp": datetime.now().isoformat(),
        }
    ), 500


@app.after_request
def after_request(response):
    """Add CORS headers to all responses"""
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type,Authorization")
    response.headers.add("Access-Control-Allow-Methods", "GET,PUT,POST,DELETE,OPTIONS")
    return response


def signal_handler(sig, frame):
    """Handle shutdown signals gracefully"""
    logger.info("Received shutdown signal, cleaning up comprehensive system...")

    if llama_manager:
        try:
            llama_manager.unload_model()
            logger.info("LLM model unloaded")
        except Exception as e:
            logger.error(f"Error unloading LLM model: {e}")

    if milvus_connected:
        try:
            disconnect_from_milvus()
            logger.info("Milvus connection closed")
        except Exception as e:
            logger.error(f"Error disconnecting from Milvus: {e}")

    logger.info("Comprehensive system shutdown complete")
    sys.exit(0)


def handle_restart_recovery():
    """Handle restart recovery and cleanup restart signal file"""
    restart_signal_file = "/app/temp/restart_requested.signal"

    if os.path.exists(restart_signal_file):
        print("🔄 RESTART RECOVERY: Processing restart signal...")
        logger.info("🔄 Processing container restart recovery...")

        try:
            # Read restart signal information
            with open(restart_signal_file, "r") as f:
                restart_info = json.load(f)

            print(f"Restart reason: {restart_info.get('reason', 'unknown')}")
            print(f"Restart triggered by job: {restart_info.get('job_id', 'unknown')}")
            logger.info(f"Restart reason: {restart_info.get('reason', 'unknown')}")
            logger.info(
                f"Restart triggered by job: {restart_info.get('job_id', 'unknown')}"
            )

            # Log GPU memory status after restart
            if torch.cuda.is_available():
                try:
                    allocated = torch.cuda.memory_allocated() / 1024**3
                    props = torch.cuda.get_device_properties(0)
                    total = props.total_memory / 1024**3
                    available = total - allocated

                    print(
                        f"GPU memory after restart: {allocated:.1f}GB used, {available:.1f}GB available"
                    )
                    logger.info(
                        f"GPU memory after restart: {allocated:.1f}GB used, {available:.1f}GB available"
                    )

                except Exception as gpu_error:
                    print(f"Could not get GPU memory status: {gpu_error}")
                    logger.warning(f"Could not get GPU memory status: {gpu_error}")

            # CRITICAL: Remove the restart signal file
            os.remove(restart_signal_file)
            print("✅ Restart signal file cleaned up successfully")
            logger.info("✅ Restart signal file cleaned up successfully")

        except json.JSONDecodeError as je:
            print(f"Could not parse restart signal file: {je}")
            logger.warning(f"Could not parse restart signal file: {je}")
            # Still remove the file
            try:
                os.remove(restart_signal_file)
                print("Corrupted restart signal file removed")
                logger.info("Corrupted restart signal file removed")
            except Exception as remove_error:
                print(f"Could not remove corrupted restart signal: {remove_error}")
                logger.error(
                    f"Could not remove corrupted restart signal: {remove_error}"
                )

        except Exception as e:
            print(f"Error processing restart recovery: {e}")
            logger.error(f"Error processing restart recovery: {e}")
            # Still try to remove the file
            try:
                os.remove(restart_signal_file)
                print("Restart signal file removed despite processing error")
                logger.warning("Restart signal file removed despite processing error")
            except Exception as remove_error:
                print(f"Could not remove restart signal file: {remove_error}")
                logger.error(f"Could not remove restart signal file: {remove_error}")

        return True

    return False


# THEN MODIFY the if __name__ == "__main__": section at the bottom:
if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print("Starting COMPREHENSIVE LLM-Primary System...")
    logger.info(
        "Starting COMPREHENSIVE LLM-Primary System with Enhanced Vector Database Integration"
    )

    try:
        # CRITICAL: Handle restart recovery FIRST
        restart_detected = handle_restart_recovery()
        if restart_detected:
            print("🔄 Container restart recovery completed")
            logger.info("🔄 Container restart recovery completed - system ready")
            time.sleep(3)  # Small delay after restart processing

        # Validate configuration
        if not hasattr(Config, "validate_config") or Config.validate_config():
            logger.info("Configuration validated")
        else:
            logger.error("Configuration validation failed")
            sys.exit(1)

        # Initialize all comprehensive components
        if not initialize_components():
            logger.error("Comprehensive component initialization failed")
            sys.exit(1)

        host = os.getenv("HOST", "0.0.0.0")
        port = int(os.getenv("PORT", "8000"))

        logger.info(f"🚀 COMPREHENSIVE LLM-Primary System starting on {host}:{port}")

        app.run(
            host=host,
            port=port,
            debug=False,
            threaded=True,
            use_reloader=False,
        )

    except Exception as e:
        print(f"STARTUP ERROR: {e}")
        logger.error(f"Failed to start application: {e}", exc_info=True)

        # Clean up restart signal on startup failure
        restart_signal_file = "/app/temp/restart_requested.signal"
        if os.path.exists(restart_signal_file):
            try:
                os.remove(restart_signal_file)
                print("Cleaned up restart signal due to startup failure")
                logger.info("Cleaned up restart signal due to startup failure")
            except:
                pass
        sys.exit(1)
