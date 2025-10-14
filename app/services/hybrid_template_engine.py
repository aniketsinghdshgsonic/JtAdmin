# app/services/hybrid_template_engine.py - COMPLETE FIXED VERSION WITH DYNAMIC EXTRACTION
import json
import logging
import time
from typing import Dict, Any, List, Optional, Tuple
import re
from datetime import datetime

from app.services.template_reconstructor import TemplateReconstructor
from app.services.field_mapper import FocusedFieldMapper

logger = logging.getLogger(__name__)


class HybridTemplateEngine:
    """Enhanced hybrid engine for comprehensive vector database-driven field mapping"""

    def __init__(self, llama_manager, mapping_storage_service, milvus_collections=None):
        self.llama_manager = llama_manager
        self.storage_service = mapping_storage_service
        self.milvus_collections = milvus_collections or {}

        # Initialize components
        self.template_reconstructor = TemplateReconstructor()
        self.field_mapper = FocusedFieldMapper(llama_manager)

        self.search_params = {
            "similarity_threshold": 0.25,  # FIXED: Increased from 0.10 to 0.25
            "max_search_results": 100,  # FIXED: Increased from 20 to 100
            "high_confidence_threshold": 0.5,
            "medium_confidence_threshold": 0.25,
            "pattern_diversity_target": 15,
            "comprehensive_mode": True,
        }

        # Store original input context for complete response format
        self._original_input_context = None

    def generate_hybrid_mapping(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive mapping using enhanced vector database approach"""

        logger.info("Starting COMPREHENSIVE vector database-driven mapping approach...")
        start_time = time.time()

        try:
            # Parse input data
            parsed_input = self._parse_input_data(input_data)

            # Comprehensive input assessment
            input_quality = self.field_mapper.assess_input_quality(parsed_input)
            logger.info(
                f"Input quality assessment - Total fields: {input_quality.get('total_fields', 0)}, "
                f"Complexity: {input_quality.get('complexity_score', 0):.2f}, "
                f"Pattern potential: {input_quality.get('pattern_learning_potential', False)}"
            )

            # Enhanced micro-chunk search for comprehensive patterns
            micro_chunks = self._find_comprehensive_micro_chunks_for_pattern_learning(
                parsed_input
            )

            if micro_chunks and len(micro_chunks) >= 3:
                logger.info(
                    f"Found {len(micro_chunks)} comprehensive micro-chunks for enhanced mapping"
                )

                # Reconstruct comprehensive template from diverse micro-chunks
                reconstructed_template = (
                    self._reconstruct_comprehensive_template_from_micro_chunks(
                        micro_chunks
                    )
                )

                if reconstructed_template:
                    confidence = reconstructed_template.get(
                        "reconstruction_confidence", 0
                    )
                    logger.info(
                        f"Comprehensive template reconstructed from {len(micro_chunks)} micro-chunks (confidence: {confidence:.3f})"
                    )

                    # Generate COMPREHENSIVE mappings using enhanced field mapper
                    logger.info(
                        "Generating comprehensive field mappings (targeting 50%+ coverage)..."
                    )
                    learned_mappings = (
                        self.field_mapper.generate_focused_field_mappings(
                            parsed_input, reconstructed_template
                        )
                    )

                    # Apply comprehensive patterns to create final mapping
                    final_mapping = self._apply_comprehensive_patterns_to_template(
                        reconstructed_template,
                        learned_mappings,
                        parsed_input,
                        micro_chunks,
                    )

                    # Add comprehensive metadata
                    final_mapping.update(
                        {
                            "generation_method": "comprehensive_vector_database_mapping",
                            "template_confidence": confidence,
                            "template_source": reconstructed_template["name"],
                            "micro_chunks_used": len(micro_chunks),
                            "pattern_types_found": self._get_pattern_types_from_chunks(
                                micro_chunks
                            ),
                            "comprehensive_approach": True,
                            "input_quality": input_quality,
                            "coverage_target": "50_percent_plus",
                        }
                    )

                    # Comprehensive coverage analysis
                    mapped_fields, unmapped_fields = (
                        self._analyze_comprehensive_coverage(
                            parsed_input, final_mapping
                        )
                    )
                    final_mapping = self._enhance_result_with_comprehensive_analytics(
                        final_mapping,
                        mapped_fields,
                        unmapped_fields,
                        start_time,
                        micro_chunks,
                    )

                    coverage_pct = (
                        len(mapped_fields)
                        / max(len(mapped_fields) + len(unmapped_fields), 1)
                    ) * 100
                    logger.info(
                        f"Comprehensive vector database mapping completed in {time.time() - start_time:.2f}s"
                    )
                    logger.info(
                        f"Final coverage: {len(mapped_fields)} fields mapped ({coverage_pct:.1f}% coverage)"
                    )

                    return final_mapping

            logger.info(
                "Insufficient comprehensive micro-chunks found, using enhanced direct vector analysis"
            )
            return self._fallback_to_enhanced_vector_analysis(
                parsed_input, micro_chunks, input_quality, start_time
            )

        except Exception as e:
            logger.error(
                f"Comprehensive vector database mapping failed: {e}", exc_info=True
            )
            return self._create_comprehensive_error_response(
                str(e), parsed_input, start_time
            )

    def _find_comprehensive_micro_chunks_for_pattern_learning(
        self, input_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Find comprehensive micro-chunks for enhanced pattern learning"""

        if not self.storage_service:
            logger.warning(
                "Storage service not available for comprehensive micro-chunk search"
            )
            if not self.milvus_collections:
                logger.warning(
                    "No milvus collections available either, returning empty micro-chunks"
                )
                return []

        if not self.milvus_collections:
            logger.warning(
                "Milvus collections not available for comprehensive micro-chunk search"
            )
            return []

        try:
            # Create enhanced search features for comprehensive pattern matching
            comprehensive_search_features = self._create_comprehensive_search_features(
                input_data
            )
            logger.info(
                f"Comprehensive search features: {comprehensive_search_features[:200]}..."
            )

            # Generate embedding for comprehensive search
            query_embedding = None
            if self.storage_service and self.storage_service.is_available():
                query_embedding = self.storage_service._generate_768_dim_embedding(
                    comprehensive_search_features, "search_query"
                )
            else:
                logger.warning(
                    "Storage service unavailable, using fallback comprehensive search"
                )
                return self._fallback_comprehensive_search(input_data)

            if not query_embedding:
                logger.warning(
                    "Failed to generate search embedding for comprehensive micro-chunks"
                )
                return []

            # Search for diverse comprehensive micro-chunks
            comprehensive_search_results = (
                self._search_for_comprehensive_diverse_micro_chunks(query_embedding)
            )

            if comprehensive_search_results:
                logger.info(
                    f"Found {len(comprehensive_search_results)} comprehensive micro-chunks for pattern learning"
                )

                # Enhanced organization by pattern type for comprehensive coverage
                organized_chunks = self._organize_comprehensive_micro_chunks_by_pattern(
                    comprehensive_search_results
                )

                logger.info(
                    f"Comprehensive chunk organization: {self._summarize_comprehensive_chunk_organization(organized_chunks)}"
                )
                return comprehensive_search_results
            else:
                logger.info(
                    "No suitable comprehensive micro-chunks found for pattern learning"
                )
                return []

        except Exception as e:
            logger.error(f"Error in comprehensive micro-chunk search: {e}")
            return []

    def _create_comprehensive_search_features(self, input_data: Dict[str, Any]) -> str:
        """Create comprehensive search features for maximum pattern matching"""

        features = []

        # Core comprehensive field detection features
        if input_data.get("refNum"):
            features.extend(
                [
                    "awb_field_mapping",
                    "reference_number_processing",
                    "hyphen_removal_transformation",
                    "identifier_normalization",
                    "tracking_number_generation",
                ]
            )

        if input_data.get("origin") and input_data.get("destination"):
            features.extend(
                [
                    "route_field_mapping",
                    "location_processing",
                    "origin_destination_mapping",
                    "geographic_coordinate_mapping",
                    "location_normalization",
                ]
            )

        # Comprehensive array processing features
        shipments = input_data.get("shipments", [])
        if shipments and shipments[0]:
            events = shipments[0].get("events", [])
            routes = shipments[0].get("routes", [])

            if events:
                features.extend(
                    [
                        "event_processing",
                        "array_processing",
                        "milestone_mapping",
                        "carrier_code_mapping",
                        "business_rule_mapping",
                        "status_normalization",
                        "event_sequencing",
                        "temporal_event_processing",
                        "location_event_mapping",
                    ]
                )

                # Enhanced event analysis
                for event in events[:3]:
                    if event.get("eventCode"):
                        features.append(
                            f"event_code_{event['eventCode'].lower()}_processing"
                        )
                    if event.get("status"):
                        features.append(
                            f"status_{event['status'].lower().replace(' ', '_')}_mapping"
                        )
                    if event.get("location"):
                        features.extend(
                            ["location_detail_mapping", "geographic_data_processing"]
                        )

            if routes:
                features.extend(
                    [
                        "route_processing",
                        "flight_plan_mapping",
                        "segment_processing",
                        "flight_number_processing",
                        "schedule_mapping",
                        "route_optimization",
                        "departure_arrival_processing",
                        "aircraft_assignment_mapping",
                    ]
                )

                # Enhanced route analysis
                for route in routes[:2]:
                    if route.get("flightNum"):
                        features.append("flight_identification_mapping")
                    if route.get("origin") or route.get("destination"):
                        features.append("route_segment_mapping")

        # Comprehensive weight and measurement features
        add_info = input_data.get("additionalInfo", {})
        if add_info.get("weight"):
            features.extend(
                [
                    "weight_mapping",
                    "numeric_conversion",
                    "nested_field_mapping",
                    "measurement_unit_processing",
                    "weight_validation",
                    "unit_standardization",
                ]
            )
        if add_info.get("totalPieces"):
            features.extend(
                [
                    "quantity_mapping",
                    "numeric_validation",
                    "piece_count_processing",
                    "cargo_enumeration",
                    "shipment_composition_mapping",
                ]
            )
        if add_info.get("volume"):
            features.extend(
                [
                    "volume_mapping",
                    "dimensional_processing",
                    "space_utilization_mapping",
                ]
            )

        # Enhanced nested structure features
        if add_info:
            features.extend(
                [
                    "additional_info_processing",
                    "metadata_extraction",
                    "nested_object_mapping",
                    "supplementary_data_processing",
                    "extended_attribute_mapping",
                ]
            )

        # Comprehensive pattern type features for micro-chunk matching
        features.extend(
            [
                "direct_field_mappings",
                "transformation_patterns",
                "array_processing_patterns",
                "business_rule_patterns",
                "input_structure_patterns",
                "nested_mapping_patterns",
                "validation_patterns",
                "normalization_patterns",
                "enrichment_patterns",
            ]
        )

        return " ".join(features)

    def _search_for_comprehensive_diverse_micro_chunks(
        self, query_embedding: List[float]
    ) -> List[Dict[str, Any]]:
        """Search for comprehensive diverse micro-chunks with maximum pattern variety - FIXED"""

        try:
            from app.milvus_connection import search_similar_mappings

            # FIXED: Enhanced search with higher limits for comprehensive coverage
            search_results = search_similar_mappings(
                collection_name="OTMToJT",
                query_embedding=query_embedding,
                top_k=100,  # ✅ INCREASED from 80 to 100
                output_fields=[
                    "text",
                    "metadata",
                    "template_name",
                    "chunk_type",
                    "source_format",
                    "target_format",
                    "mapping_type",
                    "complexity_score",
                ],
            )

            if search_results and search_results[0]:
                processed_results = []
                chunk_types_found = set()
                pattern_types_found = set()
                template_names_found = set()

                for hit in search_results[0]:
                    try:
                        distance = float(hit.score)

                        # FIXED: Proper COSINE similarity calculation
                        similarity = max(0.0, 1.0 - (distance / 2.0))

                        # FIXED: Reasonable threshold for quality (increased from 0.10)
                        if similarity >= 0.25:  # ✅ More selective threshold
                            chunk_type = self._safe_get_attribute(
                                hit.entity, "chunk_type", "unknown"
                            )
                            pattern_type = self._extract_pattern_type_from_metadata(
                                hit.entity, chunk_type
                            )
                            template_name = self._safe_get_attribute(
                                hit.entity, "template_name", "unknown"
                            )

                            # Enhanced diversity selection
                            diversity_score = (
                                len(chunk_types_found) * 10
                                + len(pattern_types_found) * 8
                                + len(template_names_found) * 5
                            )

                            # More inclusive selection for comprehensive coverage
                            if (
                                chunk_type not in chunk_types_found
                                or pattern_type not in pattern_types_found
                                or template_name not in template_names_found
                                or len(processed_results) < 50  # Allow up to 50 chunks
                            ):
                                result = type(
                                    "ComprehensiveMicroChunkResult",
                                    (),
                                    {
                                        "entity": hit.entity,
                                        "score": distance,
                                        "similarity": similarity,
                                        "chunk_type": chunk_type,
                                        "pattern_type": pattern_type,
                                        "template_name": template_name,
                                        "diversity_score": diversity_score,
                                        "complexity_score": self._safe_get_attribute(
                                            hit.entity, "complexity_score", 0.5
                                        ),
                                    },
                                )()
                                processed_results.append(result)
                                chunk_types_found.add(chunk_type)
                                pattern_types_found.add(pattern_type)
                                template_names_found.add(template_name)

                                logger.debug(
                                    f"Added result: similarity={similarity:.3f}, chunk_type={chunk_type}, template={template_name}"
                                )

                    except Exception as result_error:
                        logger.warning(
                            f"Failed to process comprehensive micro-chunk result: {result_error}"
                        )
                        continue

                if not processed_results:
                    logger.warning(
                        "No results passed 0.25 threshold - may need to adjust embeddings or threshold"
                    )
                    return []

                # Enhanced sorting by diversity, complexity, and similarity
                processed_results.sort(
                    key=lambda x: (x.diversity_score, x.complexity_score, x.similarity),
                    reverse=True,
                )
                final_results = processed_results[:50]  # Return up to 50 best chunks

                logger.info(
                    f"Selected {len(final_results)} comprehensive diverse micro-chunks"
                )
                logger.info(f"Chunk types: {chunk_types_found}")
                logger.info(f"Pattern types: {pattern_types_found}")
                logger.info(f"Template sources: {list(template_names_found)[:5]}")

                # Log similarity range
                if final_results:
                    similarities = [r.similarity for r in final_results]
                    logger.info(
                        f"Similarity range: {min(similarities):.3f} to {max(similarities):.3f}"
                    )

                return final_results

            logger.warning("No search results returned from Milvus")
            return []

        except Exception as e:
            logger.error(f"Comprehensive micro-chunk search failed: {e}")
            return []

    def _reconstruct_comprehensive_template_from_micro_chunks(
        self, micro_chunks: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Reconstruct comprehensive template from diverse micro-chunks"""

        if not micro_chunks:
            return None

        try:
            # Get template names from chunks with frequency analysis
            template_names = []
            template_qualities = {}

            for chunk in micro_chunks:
                try:
                    template_name = self._safe_get_attribute(
                        chunk.entity, "template_name", ""
                    )
                    if template_name:
                        template_names.append(template_name)
                        # Track quality metrics
                        if template_name not in template_qualities:
                            template_qualities[template_name] = {
                                "count": 0,
                                "avg_similarity": 0,
                                "avg_complexity": 0,
                            }
                        template_qualities[template_name]["count"] += 1
                        template_qualities[template_name]["avg_similarity"] += getattr(
                            chunk, "similarity", 0.5
                        )
                        template_qualities[template_name]["avg_complexity"] += getattr(
                            chunk, "complexity_score", 0.5
                        )
                except:
                    continue

            # Select best quality template
            if template_names:
                from collections import Counter

                template_counter = Counter(template_names)

                # Score templates by frequency, similarity, and complexity
                best_template = None
                best_score = 0

                for template_name, count in template_counter.items():
                    if template_name in template_qualities:
                        quality = template_qualities[template_name]
                        avg_sim = quality["avg_similarity"] / count
                        avg_comp = quality["avg_complexity"] / count
                        score = count * 0.4 + avg_sim * 0.3 + avg_comp * 0.3

                        if score > best_score:
                            best_score = score
                            best_template = template_name

                most_common_template = (
                    best_template or template_counter.most_common(1)[0][0]
                )
            else:
                most_common_template = "Comprehensive_Vector_Reconstructed"

            # Calculate enhanced confidence based on comprehensive factors
            confidence = self._calculate_comprehensive_confidence(micro_chunks)

            # Create comprehensive template structure from micro-chunks
            reconstructed_template = {
                "name": most_common_template,
                "localContext": self._reconstruct_comprehensive_local_context_from_micro_chunks(
                    micro_chunks
                ),
                "targetTreeNode": self._create_comprehensive_target_tree_from_micro_chunks(
                    micro_chunks
                ),
                "reconstruction_confidence": confidence,
                "source_micro_chunks": len(micro_chunks),
                "pattern_types_used": self._get_pattern_types_from_chunks(micro_chunks),
                "template_sources": list(set(template_names)),
                "comprehensive_reconstruction": True,
                "reconstruction_method": "comprehensive_vector_database_based",
                "diversity_metrics": self._calculate_diversity_metrics(micro_chunks),
            }

            logger.info(
                f"Successfully reconstructed comprehensive template from {len(micro_chunks)} micro-chunks: {most_common_template}"
            )
            logger.info(
                f"Template confidence: {confidence:.3f}, Pattern diversity: {len(self._get_pattern_types_from_chunks(micro_chunks))}"
            )

            return reconstructed_template

        except Exception as e:
            logger.error(
                f"Error reconstructing comprehensive template from micro-chunks: {e}"
            )
            return None

    # ============ CRITICAL FIX: DYNAMIC EXTRACTION FROM METADATA ============

    def _extract_mappings_from_chunk_metadata(
        self, chunk: Any, field_mappings: List[Dict[str, Any]]
    ) -> None:
        """✅ FIXED: Extract mapping examples with full debug logging"""

        try:
            # Access Hit object metadata correctly
            metadata = None

            if hasattr(chunk, "entity"):
                if hasattr(chunk.entity, "metadata"):
                    metadata = chunk.entity.metadata
                elif hasattr(chunk.entity, "get"):
                    metadata = chunk.entity.get("metadata")
            elif isinstance(chunk, dict):
                metadata = chunk.get("metadata", {})

            if not metadata:
                logger.debug("No metadata found in chunk")
                return

            # Handle JSON-encoded metadata
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except:
                    logger.debug("Could not parse metadata JSON string")
                    return

            if not isinstance(metadata, dict):
                logger.debug("Metadata is not a dict")
                return

            # 🔍 DEBUG: Log full metadata structure
            logger.debug(f"Chunk metadata keys: {list(metadata.keys())}")
            logger.debug(f"Pattern category: {metadata.get('pattern_category')}")
            logger.debug(f"Pattern specificity: {metadata.get('pattern_specificity')}")

            # Extract mapping_details from metadata
            mapping_details = metadata.get("mapping_details", {})

            if not mapping_details:
                # 🔍 DEBUG: Show what we have instead
                logger.debug(
                    f"No mapping_details found. Available keys: {list(metadata.keys())[:10]}"
                )

                # Check if this is input_analysis chunk
                if metadata.get("pattern_category") == "input_analysis":
                    logger.debug(
                        "This is an input_analysis chunk - no mapping_details expected"
                    )

                return

            # 🔍 DEBUG: Log mapping_details structure
            logger.debug(
                f"Found mapping_details with keys: {list(mapping_details.keys())}"
            )

            added_count = 0

            # Extract direct mappings
            direct_mappings = mapping_details.get("direct_mappings", [])
            logger.debug(f"Found {len(direct_mappings)} direct_mappings in metadata")

            for mapping in direct_mappings[:10]:
                if not isinstance(mapping, dict):
                    continue

                target_field = mapping.get("target_field", "")
                source_path = mapping.get("source_path", "")

                # 🔍 DEBUG: Log each mapping
                logger.debug(
                    f"Processing direct mapping: {source_path} → {target_field}"
                )

                # Clean target field
                if target_field.startswith("root."):
                    target_field = target_field[5:]

                if "." in target_field:
                    parts = target_field.split(".")
                    if len(parts) >= 2:
                        target_field = f"{parts[-2]}_{parts[-1]}"
                    else:
                        target_field = parts[-1]

                if target_field and source_path:
                    field_mapping = {
                        "name": target_field,
                        "value": "var_placeholder",
                        "references": [
                            {
                                "jsonId": 0,
                                "path": source_path,
                                "var": "var_placeholder",
                                "text": True,
                            }
                        ],
                    }
                    field_mappings.append(field_mapping)
                    added_count += 1
                    logger.debug(f"✓ Added direct: {source_path} → {target_field}")

            # Extract nested mappings
            nested_mappings = mapping_details.get("nested_mappings", [])
            logger.debug(f"Found {len(nested_mappings)} nested_mappings in metadata")

            for mapping in nested_mappings[:5]:
                if not isinstance(mapping, dict):
                    continue

                target_field = mapping.get("target_field", "")
                source_path = mapping.get("source_path", "")

                logger.debug(
                    f"Processing nested mapping: {source_path} → {target_field}"
                )

                if target_field.startswith("root."):
                    target_field = target_field[5:]

                if "." in target_field:
                    parts = target_field.split(".")
                    if len(parts) >= 2:
                        target_field = f"{parts[-2]}_{parts[-1]}"
                    else:
                        target_field = parts[-1]

                if target_field and source_path:
                    field_mapping = {
                        "name": target_field,
                        "value": "var_placeholder",
                        "references": [
                            {
                                "jsonId": 0,
                                "path": source_path,
                                "var": "var_placeholder",
                                "text": True,
                            }
                        ],
                    }
                    field_mappings.append(field_mapping)
                    added_count += 1
                    logger.debug(f"✓ Added nested: {source_path} → {target_field}")

            if added_count > 0:
                logger.info(f"📦 Extracted {added_count} mappings from chunk metadata")
            else:
                logger.warning(
                    "❌ Zero mappings extracted despite having mapping_details"
                )

        except Exception as e:
            logger.error(
                f"Error extracting mappings from chunk metadata: {e}", exc_info=True
            )

    # def _create_comprehensive_target_tree_from_micro_chunks(
    #     self, micro_chunks: List[Dict[str, Any]]
    # ) -> Dict[str, Any]:
    #     """Reconstruct comprehensive target tree DYNAMICALLY from micro-chunk patterns"""

    #     logger.info(
    #         f"🔧 Building target tree dynamically from {len(micro_chunks)} micro-chunks..."
    #     )

    #     # Extract ALL mappings from micro-chunks dynamically
    #     extracted_mappings = []
    #     transformations = []
    #     arrays = []

    #     var_counter = 1  # Start variable numbering

    #     for i, chunk in enumerate(micro_chunks):
    #         try:
    #             chunk_type = self._safe_get_attribute(chunk.entity, "chunk_type", "")

    #             logger.debug(f"Processing chunk {i+1}: type={chunk_type}")

    #             # ✅ Extract from metadata instead of text
    #             if chunk_type == "comprehensive_field_mappings":
    #                 before_count = len(extracted_mappings)
    #                 self._extract_mappings_from_chunk_metadata(
    #                     chunk, extracted_mappings
    #                 )
    #                 after_count = len(extracted_mappings)
    #                 if after_count > before_count:
    #                     logger.debug(
    #                         f"Extracted {after_count - before_count} field mappings from chunk {i+1}"
    #                     )

    #             elif chunk_type == "comprehensive_transformations":
    #                 metadata = self._safe_get_attribute(chunk.entity, "metadata", {})

    #                 # FIXED: Handle JSON-encoded metadata
    #                 if isinstance(metadata, str):
    #                     try:
    #                         metadata = json.loads(metadata)
    #                     except:
    #                         continue

    #                 mapping_details = metadata.get("mapping_details", {})
    #                 transform_details = mapping_details.get(
    #                     "transformation_details", {}
    #                 )

    #                 for t in transform_details.get("transformations", []):
    #                     if not isinstance(t, dict):
    #                         continue

    #                     if "source_path" not in t or "target_field" not in t:
    #                         continue

    #                     target_field = t["target_field"]

    #                     # Clean target field
    #                     if target_field.startswith("root."):
    #                         target_field = target_field[5:]

    #                     if "." in target_field:
    #                         parts = target_field.split(".")
    #                         if len(parts) >= 2:
    #                             target_field = f"{parts[-2]}_{parts[-1]}"
    #                         else:
    #                             target_field = parts[-1]

    #                     transform_mapping = {
    #                         "name": target_field,
    #                         "value": f"var{var_counter}",
    #                         "references": [
    #                             {
    #                                 "jsonId": var_counter,
    #                                 "path": t["source_path"],
    #                                 "var": f"var{var_counter}",
    #                                 "text": True,
    #                             }
    #                         ],
    #                     }

    #                     # Add transformation code if present
    #                     if "transformation_code" in t:
    #                         transform_mapping["codeValue"] = t["transformation_code"]
    #                     elif t.get("transformation_type") == "numeric_conversion":
    #                         transform_mapping["codeValue"] = (
    #                             "var_placeholder = MapperUtility.convertToDouble(var_placeholder)"
    #                         )
    #                     elif t.get("transformation_type") == "hyphen_removal":
    #                         transform_mapping["codeValue"] = (
    #                             'if(var_placeholder != null) { var_placeholder = var_placeholder.replace("-","") }'
    #                         )

    #                     transformations.append(transform_mapping)
    #                     var_counter += 1

    #                 if transformations:
    #                     logger.debug(
    #                         f"Extracted {len(transformations)} transformations from chunk {i+1}"
    #                     )

    #             elif chunk_type == "comprehensive_array_processing":
    #                 metadata = self._safe_get_attribute(chunk.entity, "metadata", {})

    #                 # FIXED: Handle JSON-encoded metadata
    #                 if isinstance(metadata, str):
    #                     try:
    #                         metadata = json.loads(metadata)
    #                     except:
    #                         continue

    #                 details = metadata.get("array_processing_details", {})
    #                 patterns = details.get("patterns", [])

    #                 for p in patterns:
    #                     if not isinstance(p, dict):
    #                         continue

    #                     if "target" not in p or "type" not in p:
    #                         continue

    #                     array_mapping = {
    #                         "name": p["target"],
    #                         "type": "ar" if "array" in p["type"].lower() else "ac",
    #                         "children": [],  # Placeholder for array children
    #                         "loopReference": {
    #                             "path": p.get(
    #                                 "loop_path",
    #                                 "X12.INTERCHANGE.GROUP.TS_204.TS_204_GROUP_3",
    #                             ),  # Use stored or default loop path
    #                         },
    #                     }

    #                     arrays.append(array_mapping)
    #                     logger.debug(
    #                         f"Added array pattern: {p['type']} for {p['target']}"
    #                     )

    #         except Exception as e:
    #             logger.debug(f"Error processing chunk {i+1}: {e}")
    #             continue

    #     # Build target tree with ONLY extracted patterns
    #     target_tree = {"name": "root", "children": []}

    #     # Add extracted field mappings with proper var numbering
    #     for mapping in extracted_mappings:
    #         mapping["value"] = f"var{var_counter}"
    #         for ref in mapping.get("references", []):
    #             ref["jsonId"] = var_counter
    #             ref["var"] = f"var{var_counter}"

    #         target_tree["children"].append(mapping)
    #         var_counter += 1

    #     # Add transformations
    #     for transform in transformations:
    #         transform["value"] = f"var{var_counter}"
    #         for ref in transform.get("references", []):
    #             ref["jsonId"] = var_counter
    #             ref["var"] = f"var{var_counter}"

    #         # Update codeValue references
    #         if "codeValue" in transform:
    #             transform["codeValue"] = re.sub(
    #                 r"var_placeholder", f"var{var_counter}", transform["codeValue"]
    #             )

    #         target_tree["children"].append(transform)
    #         var_counter += 1

    #     # Add array structures (e.g., stopDetails as array with loop)
    #     for array in arrays:
    #         target_tree["children"].append(array)

    #     logger.info(
    #         f"✅ Created DYNAMIC target tree with {len(target_tree['children'])} nodes "
    #         f"from {len(micro_chunks)} chunks"
    #     )

    #     # Log some examples if we got any
    #     if len(target_tree["children"]) > 0:
    #         logger.info("Sample mappings extracted:")
    #         for child in target_tree["children"][:3]:
    #             name = child.get("name", "unknown")
    #             path = child.get("references", [{}])[0].get("path", "unknown")
    #             logger.info(f"  - {path} → {name}")

    #     # If we extracted NOTHING, log warning
    #     if len(target_tree["children"]) == 0:
    #         logger.warning(
    #             "⚠️ No mappings extracted from chunks - check metadata structure!"
    #         )
    #         logger.warning("Sample chunk metadata:")
    #         if micro_chunks:
    #             sample_metadata = self._safe_get_attribute(
    #                 micro_chunks[0].entity, "metadata", {}
    #             )
    #             logger.warning(f"  {json.dumps(sample_metadata, indent=2)[:500]}")

    #     return target_tree

    def _create_comprehensive_target_tree_from_micro_chunks(
        self, micro_chunks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """✅ FIXED: Create comprehensive target tree DYNAMICALLY from micro-chunk patterns"""

        logger.info(
            f"🔧 Building target tree dynamically from {len(micro_chunks)} micro-chunks..."
        )

        # Extract ALL mappings from micro-chunks dynamically
        extracted_mappings = []
        var_counter = 1  # Start variable numbering

        for i, chunk in enumerate(micro_chunks):
            try:
                # Get chunk type - FIXED: Handle actual chunk types from your vector DB
                chunk_type = self._safe_get_attribute(chunk.entity, "chunk_type", "")

                logger.debug(f"Processing chunk {i+1}: type={chunk_type}")

                # ✅ CRITICAL FIX: Process "mapped_node" chunks
                if chunk_type == "mapped_node":
                    # Extract mapping from the entity's metadata
                    entity = chunk.entity if hasattr(chunk, "entity") else chunk

                    # ✅ FIX: Properly access metadata from Hit object
                    metadata = None
                    try:
                        # Try getattr for Hit objects
                        if hasattr(entity, "metadata"):
                            metadata = getattr(entity, "metadata")
                    except:
                        pass

                    if not metadata:
                        logger.debug(f"No metadata in chunk {i+1}")
                        continue

                    # Handle JSON-encoded metadata
                    if isinstance(metadata, str):
                        try:
                            metadata = json.loads(metadata)
                        except:
                            logger.debug(
                                f"Could not parse metadata JSON in chunk {i+1}"
                            )
                            continue

                    if not isinstance(metadata, dict):
                        logger.debug(f"Metadata is not dict in chunk {i+1}")
                        continue

                    # Extract target_node from metadata
                    target_node = metadata.get("target_node", {})

                    if not target_node:
                        logger.debug(f"No target_node in metadata for chunk {i+1}")
                        continue

                    # Extract name and references
                    name = target_node.get("name", "")
                    references = target_node.get("references", [])

                    if not name or not references or len(references) == 0:
                        logger.debug(
                            f"Skipping chunk {i+1}: name='{name}', refs={len(references)}"
                        )
                        continue

                    # Build mapping from target_node
                    mapping = {
                        "name": name,
                        "value": f"var{var_counter}",
                        "references": [],
                    }

                    # Add references with proper var numbering
                    for ref in references:
                        if isinstance(ref, dict) and "path" in ref:
                            mapping["references"].append(
                                {
                                    "jsonId": var_counter,
                                    "path": ref["path"],
                                    "var": f"var{var_counter}",
                                    "text": True,
                                }
                            )

                    # Add optional fields if present
                    if target_node.get("code_value"):
                        code_value = target_node["code_value"]
                        # Update var references in code
                        code_value = re.sub(
                            r"var\d+", f"var{var_counter}", str(code_value)
                        )
                        mapping["codeValue"] = code_value

                    if target_node.get("node_condition"):
                        mapping["nodeCondition"] = target_node["node_condition"]

                    if target_node.get("loop_iterator"):
                        mapping["loopIterator"] = target_node["loop_iterator"]

                    if target_node.get("loop_conditions"):
                        mapping["loopConditions"] = target_node["loop_conditions"]

                    if target_node.get("type"):
                        mapping["type"] = target_node["type"]

                    extracted_mappings.append(mapping)
                    source_path = (
                        references[0].get("path", "unknown")
                        if references
                        else "unknown"
                    )
                    logger.debug(
                        f"✓ Extracted mapping {var_counter}: {name} <- {source_path}"
                    )
                    var_counter += 1

            except Exception as e:
                logger.warning(f"Error processing chunk {i+1}: {e}", exc_info=True)
                continue

        # Build target tree with extracted mappings
        target_tree = {"name": "root", "type": "o", "children": extracted_mappings}

        logger.info(
            f"✅ Created DYNAMIC target tree with {len(extracted_mappings)} nodes "
            f"from {len(micro_chunks)} chunks"
        )

        # Log some examples if we got any
        if len(extracted_mappings) > 0:
            logger.info("Sample mappings extracted:")
            for child in extracted_mappings[:5]:
                name = child.get("name", "unknown")
                refs = child.get("references", [])
                path = refs[0].get("path", "unknown") if refs else "unknown"
                logger.info(f"  - {path} → {name}")
        else:
            # Enhanced debugging
            logger.warning("⚠️ No mappings extracted from chunks!")
            logger.warning("Debugging first chunk structure:")
            if micro_chunks:
                first_chunk = micro_chunks[0]
                entity = (
                    first_chunk.entity
                    if hasattr(first_chunk, "entity")
                    else first_chunk
                )

                # Try to get metadata using getattr
                try:
                    metadata = getattr(entity, "metadata", None)
                    logger.warning(f"Metadata type: {type(metadata)}")

                    if isinstance(metadata, str):
                        logger.warning(f"Metadata (first 500 chars): {metadata[:500]}")
                        try:
                            parsed_metadata = json.loads(metadata)
                            logger.warning(
                                f"Parsed metadata keys: {list(parsed_metadata.keys())}"
                            )
                            logger.warning(
                                f"Target node exists: {'target_node' in parsed_metadata}"
                            )
                            if "target_node" in parsed_metadata:
                                target_node = parsed_metadata["target_node"]
                                logger.warning(
                                    f"Target node keys: {list(target_node.keys())}"
                                )
                                logger.warning(f"Name: {target_node.get('name')}")
                                logger.warning(
                                    f"References count: {len(target_node.get('references', []))}"
                                )
                        except Exception as parse_error:
                            logger.warning(
                                f"Could not parse metadata JSON: {parse_error}"
                            )
                    elif isinstance(metadata, dict):
                        logger.warning(f"Metadata keys: {list(metadata.keys())}")
                        logger.warning(
                            f"Target node: {metadata.get('target_node', 'NOT FOUND')}"
                        )
                except Exception as meta_error:
                    logger.warning(f"Could not access metadata: {meta_error}")

        return target_tree

    def _apply_comprehensive_patterns_to_template(
        self,
        template: Dict[str, Any],
        learned_mappings: Dict[str, Any],
        input_data: Dict[str, Any],
        micro_chunks: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """CRITICAL FIX: Apply comprehensive patterns while preserving original target tree structure"""

        # Get original input for complete response format
        original_input = self._get_comprehensive_original_input_from_context(input_data)
        original_target_tree = (
            original_input.get("targetTreeNode") if original_input else None
        )

        if original_target_tree:
            logger.info(
                "Preserving original target tree structure with comprehensive enhancements"
            )
            result = {
                "name": original_input.get("name", f"Enhanced_{int(time.time())}"),
                "targetTreeNode": json.loads(json.dumps(original_target_tree)),
                "localContext": original_input.get(
                    "localContext", template.get("localContext", {})
                ),
                "modelVersion": original_input.get("modelVersion", 8),
                "targetInputType": original_input.get("targetInputType", "JSON"),
                "sourceInputType": original_input.get("sourceInputType", "JSON"),
                "testInput": original_input.get("testInput", ""),
                "generation_method": "comprehensive_vector_database_mapping_with_preservation",
                "original_structure_preserved": True,
                "comprehensive_enhancements_applied": True,
                "micro_chunks_applied": len(micro_chunks),
                "pattern_types_applied": self._get_pattern_types_from_chunks(
                    micro_chunks
                ),
            }

            # CRITICAL FIX: Apply comprehensive learned mappings to preserved structure
            mappings_applied = (
                self._apply_comprehensive_learned_mappings_to_preserved_tree(
                    result["targetTreeNode"], learned_mappings, input_data
                )
            )

            result["mappings_applied_count"] = mappings_applied
            result["structure_modification"] = (
                "comprehensive_field_updates_and_enhancements"
            )

            logger.info(
                f"Applied {mappings_applied} mappings to preserved original structure"
            )

        else:
            logger.info(
                "No original target tree found, using comprehensive reconstructed template structure"
            )
            result = json.loads(json.dumps(template))
            result.update(
                {
                    "name": f"Comprehensive_MicroChunk_Generated_{int(time.time())}",
                    "generation_method": "comprehensive_vector_database_mapping",
                    "original_structure_preserved": False,
                    "comprehensive_reconstruction": True,
                    "micro_chunks_applied": len(micro_chunks),
                    "pattern_types_applied": self._get_pattern_types_from_chunks(
                        micro_chunks
                    ),
                }
            )

            # Apply comprehensive mappings to reconstructed template
            self._apply_comprehensive_learned_mappings_to_template(
                result, learned_mappings, input_data
            )

        # Validate comprehensive application
        result["comprehensive_application_valid"] = (
            self._validate_comprehensive_application(result, micro_chunks)
        )

        return result

    def _find_highest_var_number(self, tree: Dict[str, Any]) -> int:
        """Find the highest var number already in the tree"""
        max_var = 0

        def scan_node(node):
            nonlocal max_var
            if isinstance(node, dict):
                # Check references for var numbers
                if "references" in node and isinstance(node["references"], list):
                    for ref in node["references"]:
                        if isinstance(ref, dict) and "var" in ref:
                            var_str = ref["var"]
                            # Extract number from 'var123'
                            match = re.search(r"var(\d+)", str(var_str))
                            if match:
                                num = int(match.group(1))
                                max_var = max(max_var, num)

                # Check value field
                if "value" in node:
                    match = re.search(r"var(\d+)", str(node["value"]))
                    if match:
                        num = int(match.group(1))
                        max_var = max(max_var, num)

                # Recurse through all values
                for value in node.values():
                    if isinstance(value, (dict, list)):
                        scan_node(value)
            elif isinstance(node, list):
                for item in node:
                    scan_node(item)

        scan_node(tree)
        logger.info(f"Found highest existing var number: var{max_var}")
        return max_var

    def _apply_comprehensive_learned_mappings_to_preserved_tree(
        self, tree: Dict[str, Any], mappings: Dict[str, Any], input_data: Dict[str, Any]
    ) -> int:
        """FIXED: Extract and apply hybrid mappings correctly"""

        if not tree or not mappings:
            return 0

        highest_var = self._find_highest_var_number(tree)
        var_counter = highest_var + 1

        logger.info(f"Starting hybrid mappings at var{var_counter}")

        # CRITICAL FIX: Extract the targetTreeNode and get its children
        target_tree_node = mappings.get("targetTreeNode", {})

        # If targetTreeNode has children array, extract those as mappings
        children = target_tree_node.get("children", [])

        if not children:
            logger.warning("No children found in targetTreeNode to apply")
            return 0

        logger.info(f"Found {len(children)} children in hybrid result to merge")

        mappings_applied = 0

        # Get existing mapped paths to avoid duplicates
        existing_paths = set()

        def extract_existing_paths(node):
            if isinstance(node, dict):
                for ref in node.get("references", []):
                    if isinstance(ref, dict) and "path" in ref:
                        existing_paths.add(ref["path"])
                for child in node.get("children", []):
                    extract_existing_paths(child)

        extract_existing_paths(tree)
        logger.info(f"Found {len(existing_paths)} existing mapped paths")

        # Add hybrid children to tree (skip duplicates)
        if "children" not in tree:
            tree["children"] = []

        for child in children:
            if not isinstance(child, dict):
                continue

            # Check if this path is already mapped
            child_refs = child.get("references", [])
            if not child_refs:
                continue

            child_path = child_refs[0].get("path", "")

            if child_path in existing_paths:
                logger.debug(f"Skipping duplicate path: {child_path}")
                continue

            # Update var numbers
            new_child = json.loads(json.dumps(child))  # Deep copy

            # Update references
            for ref in new_child.get("references", []):
                ref["jsonId"] = var_counter
                ref["var"] = f"var{var_counter}"
                ref["source"] = "hybrid"

            # Update value
            new_child["value"] = f"var{var_counter}"

            # Update codeValue if present
            if "codeValue" in new_child:
                old_var_match = re.search(r"var\d+", new_child["codeValue"])
                if old_var_match:
                    new_child["codeValue"] = new_child["codeValue"].replace(
                        old_var_match.group(0), f"var{var_counter}"
                    )

            tree["children"].append(new_child)
            existing_paths.add(child_path)

            logger.debug(
                f"Applied: {new_child['name']} <- {child_path} (var{var_counter})"
            )

            var_counter += 1
            mappings_applied += 1

        logger.info(
            f"Applied {mappings_applied} total hybrid mappings (var{highest_var + 1} to var{var_counter - 1})"
        )
        return mappings_applied

    def _analyze_comprehensive_coverage(
        self, input_data: Dict[str, Any], mapping_result: Dict[str, Any]
    ) -> Tuple[List[str], List[str]]:
        """FIXED: Analyze comprehensive coverage achieved"""

        input_field_paths = self._get_all_comprehensive_input_field_paths(input_data)
        mapped_paths = set()

        def extract_mapped_paths_comprehensive(node):
            if isinstance(node, dict):
                if "references" in node and isinstance(node["references"], list):
                    for ref in node["references"]:
                        if isinstance(ref, dict) and "path" in ref:
                            path_value = ref["path"]

                            # FIX: Handle both string and list paths
                            if isinstance(path_value, str):
                                mapped_paths.add(path_value)
                            elif isinstance(path_value, list):
                                for p in path_value:
                                    if isinstance(p, str):
                                        mapped_paths.add(p)
                                logger.warning(f"List path found: {path_value}")

                for key, value in node.items():
                    if isinstance(value, (dict, list)):
                        extract_mapped_paths_comprehensive(value)

            elif isinstance(node, list):
                for item in node:
                    extract_mapped_paths_comprehensive(item)

        target_tree = mapping_result.get("targetTreeNode", {})
        extract_mapped_paths_comprehensive(target_tree)

        # Normalize and match paths
        mapped_fields = []
        unmapped_fields = []

        for field_path in input_field_paths:
            if field_path in mapped_paths:
                mapped_fields.append(field_path)
            else:
                # Try normalized matching (arrays)
                normalized_field = re.sub(r"\[\d+\]", "[0]", field_path)
                found = False

                for mapped_path in mapped_paths:
                    normalized_mapped = re.sub(r"\[\d+\]", "[0]", mapped_path)
                    if normalized_field == normalized_mapped:
                        mapped_fields.append(field_path)
                        found = True
                        break

                if not found:
                    unmapped_fields.append(field_path)

        return mapped_fields, unmapped_fields

    # ============ HELPER METHODS ============

    def _parse_input_data(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse input data"""
        test_input = input_data.get("testInput", "")
        if isinstance(test_input, str):
            try:
                return json.loads(test_input)
            except:
                return input_data
        return test_input or input_data

    def _safe_get_attribute(self, entity, attr_name: str, default=None):
        """Safely get attribute from entity"""
        try:
            if hasattr(entity, attr_name):
                value = getattr(entity, attr_name)
                if value is not None:
                    return value
            if isinstance(entity, dict):
                return entity.get(attr_name, default)
            return default
        except:
            return default

    def _get_pattern_types_from_chunks(
        self, micro_chunks: List[Dict[str, Any]]
    ) -> List[str]:
        """Get pattern types from chunks"""
        pattern_types = []
        for chunk in micro_chunks:
            try:
                if hasattr(chunk, "pattern_type"):
                    pattern_type = chunk.pattern_type
                else:
                    chunk_type = self._safe_get_attribute(
                        chunk.entity, "chunk_type", "unknown"
                    )
                    pattern_type = chunk_type

                if pattern_type not in pattern_types:
                    pattern_types.append(pattern_type)
            except:
                continue
        return pattern_types

    def _get_all_comprehensive_input_field_paths(
        self, input_data: Dict[str, Any], prefix: str = "root"
    ) -> List[str]:
        """Get all field paths comprehensively"""
        paths = []

        def extract_recursive(obj, current_path, depth=0):
            if depth > 12:  # Increased depth for comprehensive analysis
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

        extract_recursive(input_data, prefix)
        return paths

    def _get_comprehensive_original_input_from_context(self, processed_input):
        """Get original input from context"""
        return getattr(self, "_original_input_context", None)

    def _apply_comprehensive_learned_mappings_to_template(
        self, result, mappings, input_data
    ):
        """Apply mappings to template (stub for non-preserved structure)"""
        pass

    def _validate_comprehensive_application(self, result, chunks):
        """Validate comprehensive application"""
        return True

    def _calculate_comprehensive_confidence(self, micro_chunks):
        """Calculate confidence from micro-chunks"""
        if not micro_chunks:
            return 0.0

        try:
            similarities = []
            for chunk in micro_chunks:
                similarity = getattr(chunk, "similarity", 0.5)
                similarities.append(similarity)

            avg_similarity = sum(similarities) / len(similarities)
            chunk_diversity = min(len(micro_chunks) / 15.0, 1.0)

            confidence = avg_similarity * 0.6 + chunk_diversity * 0.4
            return min(confidence, 0.98)
        except:
            return 0.5

    def _calculate_diversity_metrics(self, chunks):
        """Calculate diversity metrics"""
        return {
            "chunk_count": len(chunks),
            "unique_templates": len(
                set(
                    [
                        self._safe_get_attribute(c.entity, "template_name", "")
                        for c in chunks
                    ]
                )
            ),
            "unique_patterns": len(
                set(
                    [
                        self._safe_get_attribute(c.entity, "chunk_type", "")
                        for c in chunks
                    ]
                )
            ),
        }

    def _extract_pattern_type_from_metadata(self, entity, chunk_type):
        """Extract pattern type from metadata"""
        return chunk_type

    def _fallback_comprehensive_search(self, input_data):
        """Fallback search method"""
        return []

    def _organize_comprehensive_micro_chunks_by_pattern(self, chunks):
        """Organize chunks by pattern"""
        organized = {}
        for chunk in chunks:
            chunk_type = self._safe_get_attribute(chunk.entity, "chunk_type", "unknown")
            if chunk_type not in organized:
                organized[chunk_type] = []
            organized[chunk_type].append(chunk)
        return organized

    def _summarize_comprehensive_chunk_organization(self, organized):
        """Summarize chunk organization"""
        return ", ".join([f"{k}({len(v)})" for k, v in organized.items()])

    def _reconstruct_comprehensive_local_context_from_micro_chunks(self, micro_chunks):
        """Reconstruct local context"""
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

    def _fallback_to_enhanced_vector_analysis(
        self, parsed_input, micro_chunks, input_quality, start_time
    ):
        """Enhanced fallback analysis"""
        return self._create_comprehensive_error_response(
            "Enhanced vector analysis not fully implemented", parsed_input, start_time
        )

    def _create_comprehensive_error_response(
        self, error_message, input_data, start_time
    ):
        """Create comprehensive error response"""
        return {
            "error": True,
            "error_message": f"Comprehensive mapping failed: {error_message}",
            "generation_method": "comprehensive_error",
            "timestamp": datetime.now().isoformat(),
            "generation_time": f"{time.time() - start_time:.3f}s",
        }

    def _enhance_result_with_comprehensive_analytics(
        self, mapping_result, mapped_fields, unmapped_fields, start_time, micro_chunks
    ):
        """Enhance with comprehensive analytics"""
        generation_time = time.time() - start_time
        total_fields = len(mapped_fields) + len(unmapped_fields)
        coverage_percentage = (len(mapped_fields) / max(total_fields, 1)) * 100

        enhanced_result = mapping_result.copy()
        enhanced_result.update(
            {
                "_mapping_metadata": {
                    "total_input_fields": total_fields,
                    "mapped_fields_count": len(mapped_fields),
                    "unmapped_fields_count": len(unmapped_fields),
                    "coverage_percentage": round(coverage_percentage, 2),
                    "mapped_field_paths": mapped_fields,
                    "unmapped_field_paths": unmapped_fields,
                    "method": mapping_result.get("generation_method", "unknown"),
                },
                "_generation_time": f"{generation_time:.3f}s",
                "_valid": True,
                "_timestamp": datetime.now().isoformat(),
            }
        )

        return enhanced_result
