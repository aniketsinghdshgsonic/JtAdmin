
# app/services/llm_primary_mapping_service.py - COMPLETE FILE
import json
import logging
import time
import re
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import torch

from app.services.hybrid_template_engine import HybridTemplateEngine

logger = logging.getLogger(__name__)


class LLMPrimaryMappingService:
    """Enhanced LLM-Primary Mapping Service with Hybrid Base + LLM Enhancement"""

    def __init__(self, llama_manager, mapping_storage_service, milvus_collections=None):
        self.llama_manager = llama_manager
        self.storage_service = mapping_storage_service
        self.milvus_collections = milvus_collections or {}
        self._original_input_context = None

        # Initialize hybrid engine
        self.hybrid_engine = HybridTemplateEngine(
            llama_manager, mapping_storage_service, milvus_collections
        )

    def set_original_input_context(self, input_data: Dict[str, Any]) -> None:
        """Store complete original input context for clean response format"""
        self._original_input_context = input_data
        if hasattr(self, "hybrid_engine") and self.hybrid_engine:
            self.hybrid_engine._original_input_context = input_data
        logger.debug("Stored complete original input context")

    def generate_mapping_with_llm(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate mappings with Hybrid Base + LLM Enhancement using few-shot learning"""
        
        logger.info("Starting Hybrid Base + LLM Enhancement mapping approach...")
        start_time = time.time()

        try:
            # Clear GPU cache before starting
            self._clear_gpu_cache()
            
            # Store and parse input
            self.set_original_input_context(input_data)
            parsed_input = self._parse_input_safely(input_data)
            total_input_fields = len(self._extract_all_field_paths_safely(parsed_input))
            logger.info(f"Input parsed: {total_input_fields} fields detected")

            # Step 1: Run hybrid mapping (this is the reliable baseline)
            logger.info("Step 1: Generating hybrid base mapping...")
            hybrid_result = self.hybrid_engine.generate_hybrid_mapping(input_data)
            
            # Clear GPU after hybrid
            self._clear_gpu_cache()

            if not hybrid_result or hybrid_result.get("error"):
                logger.error("Hybrid mapping failed, cannot proceed")
                return self._create_comprehensive_error_mapping(
                    "Hybrid base mapping failed", input_data, start_time
                )

            # Analyze hybrid coverage
            hybrid_mapped_fields, hybrid_unmapped_fields = self._analyze_comprehensive_coverage(
                parsed_input, hybrid_result
            )
            hybrid_coverage_pct = (
                (len(hybrid_mapped_fields) / total_input_fields) * 100
                if total_input_fields > 0
                else 0.0
            )
            
            logger.info(
                f"Hybrid base: {len(hybrid_mapped_fields)}/{total_input_fields} fields "
                f"({hybrid_coverage_pct:.1f}% coverage)"
            )

            # If hybrid already achieved good coverage, return it
            if len(hybrid_unmapped_fields) == 0:
                logger.info("Hybrid achieved complete coverage, no LLM enhancement needed")
                hybrid_result["field_coverage_analysis"] = {
                    "mapped_fields_count": len(hybrid_mapped_fields),
                    "unmapped_fields_count": 0,
                    "coverage_percentage": round(hybrid_coverage_pct, 2),
                    "generation_method": "hybrid_only_complete",
                }
                hybrid_result.update({
                    "service_version": "llm_primary_v8.0_hybrid_complete",
                    "generation_time": f"{time.time() - start_time:.3f}s",
                    "timestamp": datetime.now().isoformat(),
                })
                return hybrid_result

            # Step 2: Get micro chunks with concrete examples
            logger.info("Step 2: Extracting concrete mapping examples from vector database...")
            
            # Get micro chunks from hybrid result or fetch new ones
            micro_chunks = self._get_micro_chunks_for_enhancement(parsed_input, hybrid_result)
            
            if not micro_chunks or len(micro_chunks) < 2:
                logger.warning("Insufficient concrete examples, using hybrid only")
                hybrid_result["field_coverage_analysis"] = {
                    "mapped_fields_count": len(hybrid_mapped_fields),
                    "unmapped_fields_count": len(hybrid_unmapped_fields),
                    "coverage_percentage": round(hybrid_coverage_pct, 2),
                    "generation_method": "hybrid_only_no_examples",
                }
                return hybrid_result

            # Step 3: Create enhanced template with micro chunks for few-shot learning
            logger.info(f"Step 3: Preparing {len(micro_chunks)} examples for LLM few-shot learning...")
            
            enhanced_template = {
                "targetTreeNode": hybrid_result.get("targetTreeNode", {}),
                "source_micro_chunks_data": micro_chunks,
                "reconstruction_confidence": hybrid_result.get("template_confidence", 0.8),
            }

            # Step 4: Use field mapper with few-shot learning
            logger.info("Step 4: Calling LLM enhancement with few-shot examples...")
            
            # Clear GPU before LLM
            self._clear_gpu_cache()
            
            from app.services.field_mapper import FocusedFieldMapper
            field_mapper = FocusedFieldMapper(self.llama_manager)
            
            enhanced_result = field_mapper.generate_focused_field_mappings(
                parsed_input, enhanced_template
            )
            
            # Clear GPU after LLM
            self._clear_gpu_cache()

            if not enhanced_result or enhanced_result.get("error"):
                logger.warning("LLM enhancement failed, using hybrid only")
                hybrid_result["field_coverage_analysis"] = {
                    "mapped_fields_count": len(hybrid_mapped_fields),
                    "unmapped_fields_count": len(hybrid_unmapped_fields),
                    "coverage_percentage": round(hybrid_coverage_pct, 2),
                    "generation_method": "hybrid_only_llm_failed",
                }
                return hybrid_result

            # Step 5: Prepare final result
            logger.info("Step 5: Analyzing final coverage...")
            
            final_mapping = {
                "name": f"Hybrid_Plus_LLM_Enhanced_{int(time.time())}",
                "targetTreeNode": enhanced_result.get("targetTreeNode", {}),
                "localContext": hybrid_result.get("localContext", {}),
            }

            # Comprehensive coverage analysis
            final_mapped_fields, final_unmapped_fields = self._analyze_comprehensive_coverage(
                parsed_input, final_mapping
            )
            final_coverage_pct = (
                (len(final_mapped_fields) / total_input_fields) * 100
                if total_input_fields > 0
                else 0.0
            )

            final_mapping.update({
                "field_coverage_analysis": {
                    "total_input_fields": total_input_fields,
                    "mapped_fields_count": len(final_mapped_fields),
                    "unmapped_fields_count": len(final_unmapped_fields),
                    "coverage_percentage": round(final_coverage_pct, 2),
                    "mapped_field_paths": final_mapped_fields,
                    "unmapped_field_paths": final_unmapped_fields,
                    "generation_method": enhanced_result.get("generation_method", "hybrid_plus_llm"),
                    "hybrid_contributed": enhanced_result.get("hybrid_contributed", len(hybrid_mapped_fields)),
                    "llm_contributed": enhanced_result.get("llm_contributed", 0),
                },
                "service_version": "llm_primary_v8.0_fewshot_enhancement",
                "generation_time": f"{time.time() - start_time:.3f}s",
                "timestamp": datetime.now().isoformat(),
                "micro_chunks_used": len(micro_chunks),
                "few_shot_learning": True,
            })

            logger.info(f"✅ Final mapping completed in {time.time() - start_time:.2f}s")
            logger.info(f"   Total coverage: {final_coverage_pct:.1f}% ({len(final_mapped_fields)}/{total_input_fields} fields)")
            logger.info(f"   Hybrid base: {len(hybrid_mapped_fields)} fields")
            logger.info(f"   LLM added: {len(final_mapped_fields) - len(hybrid_mapped_fields)} fields")

            return final_mapping

        except Exception as e:
            logger.error(f"Mapping generation failed: {str(e)}", exc_info=True)
            return self._create_comprehensive_error_mapping(str(e), input_data, start_time)

    def _get_micro_chunks_for_enhancement(
        self, parsed_input: Dict[str, Any], hybrid_result: Dict[str, Any]
    ) -> List[Any]:
        """Get micro chunks for LLM enhancement"""
        
        # Try to get from hybrid result first
        if "micro_chunks_used" in hybrid_result:
            micro_chunks = hybrid_result.get("micro_chunks_data", [])
            if micro_chunks and len(micro_chunks) > 0:
                return micro_chunks

        # Otherwise, search for new chunks
        if not self.storage_service or not self.storage_service.is_available():
            return []

        try:
            # Create search features
            search_features = self._create_comprehensive_search_features(parsed_input)
            
            # Generate embedding
            query_embedding = self.storage_service._generate_768_dim_embedding(
                search_features, "search_query"
            )
            
            if not query_embedding:
                return []

            # Search for micro chunks
            from app.milvus_connection import search_similar_mappings
            
            search_results = search_similar_mappings(
                collection_name="OTMToJT",
                query_embedding=query_embedding,
                top_k=20,
                output_fields=[
                    "text", "metadata", "template_name", "chunk_type",
                    "source_format", "target_format", "mapping_type"
                ],
            )

            if search_results and search_results[0]:
                logger.info(f"Found {len(search_results[0])} micro chunks from vector database")
                return search_results[0]

        except Exception as e:
            logger.error(f"Error fetching micro chunks: {e}")

        return []

    def _create_comprehensive_search_features(self, input_data: Dict[str, Any]) -> str:
        """Create comprehensive search features"""
        
        features = []

        if input_data.get("refNum"):
            features.extend([
                "awb_field_mapping", "reference_number_processing",
                "identifier_normalization", "tracking_number_generation"
            ])

        if input_data.get("origin") and input_data.get("destination"):
            features.extend([
                "route_field_mapping", "location_processing",
                "origin_destination_mapping"
            ])

        shipments = input_data.get("shipments", [])
        if shipments and shipments[0]:
            events = shipments[0].get("events", [])
            routes = shipments[0].get("routes", [])

            if events:
                features.extend([
                    "event_processing", "array_processing",
                    "milestone_mapping", "status_normalization"
                ])

            if routes:
                features.extend([
                    "route_processing", "flight_plan_mapping",
                    "segment_processing"
                ])

        add_info = input_data.get("additionalInfo", {})
        if add_info.get("weight"):
            features.extend(["weight_mapping", "numeric_conversion"])
        if add_info.get("totalPieces"):
            features.extend(["quantity_mapping", "piece_count_processing"])

        features.extend([
            "direct_field_mappings", "transformation_patterns",
            "business_rule_patterns"
        ])

        return " ".join(features)

    def _analyze_comprehensive_coverage(
        self, input_data: Dict[str, Any], mapping_result: Dict[str, Any]
    ) -> Tuple[List[str], List[str]]:
        """Analyze comprehensive coverage achieved"""
        
        input_field_paths = self._extract_all_field_paths_safely(input_data)
        mapped_paths = set()

        def extract_mapped_paths_comprehensive(node):
            if isinstance(node, dict):
                if "references" in node and isinstance(node["references"], list):
                    for ref in node["references"]:
                        if isinstance(ref, dict) and "path" in ref:
                            path_value = ref["path"]

                            if isinstance(path_value, str):
                                mapped_paths.add(path_value)
                            elif isinstance(path_value, list):
                                for p in path_value:
                                    if isinstance(p, str):
                                        mapped_paths.add(p)

                for key, value in node.items():
                    if isinstance(value, (dict, list)):
                        extract_mapped_paths_comprehensive(value)

            elif isinstance(node, list):
                for item in node:
                    extract_mapped_paths_comprehensive(item)

        target_tree = mapping_result.get("targetTreeNode", {})
        extract_mapped_paths_comprehensive(target_tree)

        mapped_fields = []
        unmapped_fields = []

        for field_path in input_field_paths:
            if field_path in mapped_paths:
                mapped_fields.append(field_path)
            else:
                normalized_field = re.sub(r'\[\d+\]', '[]', field_path)
                found = False
                for mapped_path in mapped_paths:
                    normalized_mapped = re.sub(r'\[\d+\]', '[]', mapped_path)
                    if normalized_field == normalized_mapped:
                        mapped_fields.append(field_path)
                        found = True
                        break

                if not found:
                    unmapped_fields.append(field_path)

        return mapped_fields, unmapped_fields

    def _parse_input_safely(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Safely parse input data"""
        try:
            test_input = input_data.get("testInput", "")
            if isinstance(test_input, str) and test_input:
                return json.loads(test_input)
            elif isinstance(test_input, dict):
                return test_input
            return input_data
        except:
            return input_data

    def _extract_all_field_paths_safely(self, input_data: Dict[str, Any]) -> List[str]:
        """Safely extract all field paths"""
        try:
            paths = []

            def extract_recursive(obj, current_path, depth=0):
                if depth > 10:
                    return
                if isinstance(obj, dict):
                    for key, value in obj.items():
                        new_path = f"{current_path}.{key}"
                        paths.append(new_path)
                        if isinstance(value, (dict, list)):
                            extract_recursive(value, new_path, depth + 1)
                elif isinstance(obj, list) and obj:
                    paths.append(f"{current_path}[0]")
                    extract_recursive(obj[0], f"{current_path}[0]", depth + 1)

            extract_recursive(input_data, "root")
            return paths
        except:
            return []

    def _create_comprehensive_error_mapping(
        self, error_message: str, input_data: Dict[str, Any], start_time: float
    ) -> Dict[str, Any]:
        """Create comprehensive error mapping"""
        return {
            "error": True,
            "error_message": f"Mapping failed: {error_message}",
            "generation_method": "error",
            "generation_time": f"{time.time() - start_time:.3f}s",
            "timestamp": datetime.now().isoformat(),
        }

    def _clear_gpu_cache(self):
        """Clear GPU cache to prevent OOM"""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.debug("GPU cache cleared")
        except Exception as e:
            logger.debug(f"Could not clear GPU cache: {e}")
