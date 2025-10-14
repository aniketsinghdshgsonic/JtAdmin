"""
UPDATED Smart Hybrid Orchestrator - Simple Text Format
LLM generates simple text mappings, Python builds JSON
Much more reliable than asking LLM to generate JSON
"""

import logging
import json
import time
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pymilvus import Collection

logger = logging.getLogger(__name__)


@dataclass
class MappingResult:
    """Result of a mapping operation"""

    success: bool
    field_path: str
    mapping: Dict[str, Any]
    confidence: float
    tier: int
    error: Optional[str] = None
    chunks_used: List[Dict] = None


class SmartHybridOrchestrator:
    """Orchestrates three-tier Smart Hybrid mapping with simple text format"""

    def __init__(
        self, llama_manager, mapping_storage_service, milvus_collections: Dict
    ):
        self.llama_manager = llama_manager
        self.storage_service = mapping_storage_service
        self.collections = milvus_collections

        # Performance tracking
        self.stats = {
            "tier1_success": 0,
            "tier1_total": 0,
            "tier2_success": 0,
            "tier2_total": 0,
            "tier3_success": 0,
            "tier3_total": 0,
            "total_time": 0.0,
            "llm_calls": 0,
        }

        # Configuration
        self.tier1_batch_size = 25
        self.tier2_parallel_workers = 1  # Sequential to prevent crashes
        self.vector_search_top_k = 7

        # Variable counter for references
        self.var_counter = 1

        logger.info("✅ Smart Hybrid Orchestrator initialized (Simple Text Format)")

    def execute_mapping(
        self,
        tier1_groups: List[Dict],  # ← ADD THESE
        tier2_fields: List[Dict],  # ← ADD THESE
        tier3_fields: List[Dict],  # ← ADD THESE
        unmapped_input: Dict[str, Any],
        target_structure: Dict[str, Any],
        source_format: str,
        target_format: str,
        collection_name: str = "OTMToJT",
    ) -> Dict[str, Any]:
        """Execute complete three-tier mapping workflow and return complete mapped template"""

        start_time = time.time()
        logger.info("🚀 Starting Smart Hybrid Mapping Workflow (Simple Text Format)")
        logger.info(f"   Source: {source_format} → {target_format}")

        # Reset variable counter
        self.var_counter = 1
        self.all_mappings = []  # ← ADD THIS to store all mappings

        results = {
            "tier1_results": [],
            "tier2_results": [],
            "tier3_results": [],
            "mapping_tree": {"name": "root", "children": []},
            "statistics": {},
            "errors": [],
        }

        try:
            # TIER 1: Batch process static/simple fields
            logger.info("=" * 60)
            logger.info("TIER 1: Batch Processing Static & Simple Fields")
            logger.info("=" * 60)

            tier1_results = self._execute_tier1(
                # field_analysis["tier_assignments"]["tier1_static_simple"],
                tier1_groups,
                source_format,
                target_format,
                collection_name,
            )
            results["tier1_results"] = tier1_results

            # ← ADD THIS: Store mappings
            self.all_mappings.extend([r for r in tier1_results if r.success])

            # TIER 2: Individual complex fields (SEQUENTIAL)
            logger.info("=" * 60)
            logger.info("TIER 2: Individual Processing Complex Fields")
            logger.info("=" * 60)

            tier2_results = self._execute_tier2(
                # field_analysis["tier_assignments"]["tier2_complex"],
                tier2_fields,
                source_format,
                target_format,
                collection_name,
            )
            results["tier2_results"] = tier2_results

            # ← ADD THIS: Store mappings
            self.all_mappings.extend([r for r in tier2_results if r.success])

            # TIER 3: Context-dependent fields
            logger.info("=" * 60)
            logger.info("TIER 3: Context-Dependent Fields")
            logger.info("=" * 60)

            # tier3_results = self._execute_tier3(
            #     field_analysis["tier_assignments"]["tier3_dependent"],
            #     results["tier1_results"] + results["tier2_results"],
            #     source_format,
            #     target_format,
            #     collection_name,
            # )
            tier3_results = self._execute_tier3(
                tier3_fields,  # ← Use parameter directly
                results["tier1_results"] + results["tier2_results"],
                source_format,
                target_format,
                collection_name,
            )
            results["tier3_results"] = tier3_results

            # ← ADD THIS: Store mappings
            self.all_mappings.extend([r for r in tier3_results if r.success])

            # ← ADD THIS ENTIRE SECTION: Build complete mapped target tree
            logger.info("🔨 Building complete mapped target tree...")
            mapped_target_tree = self._inject_mappings_into_target_tree(
                target_structure, self.all_mappings
            )

            # Calculate statistics
            total_time = time.time() - start_time
            self.stats["total_time"] = total_time

            results["statistics"] = {
                "total_fields": len(tier1_results)
                + len(tier2_results)
                + len(tier3_results),
                "successful_mappings": len(self.all_mappings),
                "tier1_accuracy": self._calculate_accuracy(tier1_results),
                "tier2_accuracy": self._calculate_accuracy(tier2_results),
                "tier3_accuracy": self._calculate_accuracy(tier3_results),
                "total_time_seconds": round(total_time, 2),
                "llm_calls": self.stats["llm_calls"],
                "avg_confidence": self._calculate_avg_confidence(
                    tier1_results + tier2_results + tier3_results
                ),
            }

            # ← ADD THIS: Include mapped tree in results
            results["mapped_target_tree"] = mapped_target_tree
            results["total_mapped"] = len(self.all_mappings)
            results["overall_accuracy"] = (
                len(self.all_mappings)
                / (len(tier1_results) + len(tier2_results) + len(tier3_results))
                * 100
                if (len(tier1_results) + len(tier2_results) + len(tier3_results)) > 0
                else 0
            )

            logger.info("=" * 60)
            logger.info("✅ MAPPING WORKFLOW COMPLETE")
            logger.info("=" * 60)
            self._log_statistics(results["statistics"])

        except Exception as e:
            logger.error(f"❌ Mapping workflow failed: {e}", exc_info=True)
            results["errors"].append(str(e))

        return results

    def _execute_tier1(
        self,
        groups: List[Dict],
        source_format: str,
        target_format: str,
        collection_name: str,
    ) -> List[MappingResult]:
        """Execute Tier 1: Batch processing with simple text format"""

        results = []

        for group_info in groups:
            group_id = group_info["group_id"]
            fields = group_info["fields"]

            logger.info(f"📦 Processing group {group_id} ({len(fields)} fields)")

            try:
                # Vector search
                group_query = self._create_group_query(fields, source_format)
                chunks = self._vector_search(group_query, collection_name, top_k=10)

                # Generate simple text mappings
                # text_mappings = self._generate_text_mappings(
                #     fields, chunks, source_format, target_format
                # )

                # if text_mappings:
                #     # Parse text and convert to JSON
                #     for field_path in fields:
                #         json_mapping = self._text_to_json(field_path, text_mappings)

                #         if json_mapping:
                #             results.append(
                #                 MappingResult(
                #                     success=True,
                #                     field_path=field_path,
                #                     mapping=json_mapping,
                #                     confidence=0.90,
                #                     tier=1,
                #                     chunks_used=chunks[:3],
                #                 )
                #             )
                #             self.stats["tier1_success"] += 1
                #         else:
                #             results.append(
                #                 MappingResult(
                #                     success=False,
                #                     field_path=field_path,
                #                     mapping={},
                #                     confidence=0.0,
                #                     tier=1,
                #                     error="Not found in text mappings",
                #                 )
                #             )

                #         self.stats["tier1_total"] += 1
                # else:
                #     logger.error(f"❌ Failed to generate text mappings for {group_id}")
                #     for field_path in fields:
                #         results.append(
                #             MappingResult(
                #                 success=False,
                #                 field_path=field_path,
                #                 mapping={},
                #                 confidence=0.0,
                #                 tier=1,
                #                 error="Text generation failed",
                #             )
                #         )
                #         self.stats["tier1_total"] += 1

                # USE CHUNKS DIRECTLY instead of generating text!
                if chunks and len(chunks) > 0:
                    logger.info(f"✅ Found {len(chunks)} similar chunks")

                    for field_path in fields:
                        # Find best matching chunk for this field
                        json_mapping = self._create_mapping_from_chunk(
                            field_path, chunks, source_format
                        )

                        if json_mapping:
                            results.append(
                                MappingResult(
                                    success=True,
                                    field_path=field_path,
                                    mapping=json_mapping,
                                    confidence=0.90,
                                    tier=1,
                                    chunks_used=chunks[:3],
                                )
                            )
                            self.stats["tier1_success"] += 1
                        else:
                            results.append(
                                MappingResult(
                                    success=False,
                                    field_path=field_path,
                                    mapping={},
                                    confidence=0.0,
                                    tier=1,
                                    error="No matching chunk found",
                                )
                            )

                        self.stats["tier1_total"] += 1

            except Exception as e:
                logger.error(f"❌ Error processing group {group_id}: {e}")
                for field_path in fields:
                    results.append(
                        MappingResult(
                            success=False,
                            field_path=field_path,
                            mapping={},
                            confidence=0.0,
                            tier=1,
                            error=str(e),
                        )
                    )
                    self.stats["tier1_total"] += 1

        logger.info(
            f"✅ Tier 1 complete: {self.stats['tier1_success']}/{self.stats['tier1_total']} successful"
        )
        return results

    def _execute_tier2(
        self,
        fields: List[Dict],
        source_format: str,
        target_format: str,
        collection_name: str,
    ) -> List[MappingResult]:
        """Execute Tier 2: SEQUENTIAL individual processing"""

        results = []

        # Process sequentially to prevent crashes
        for field_info in fields:
            field_path = field_info["field"]
            logger.info(f"⚙️ Processing complex field: {field_path}")

            try:
                result = self._process_complex_field(
                    field_info, source_format, target_format, collection_name
                )
                results.append(result)

                if result.success:
                    self.stats["tier2_success"] += 1
                self.stats["tier2_total"] += 1

            except Exception as e:
                logger.error(f"❌ Error processing {field_path}: {e}")
                results.append(
                    MappingResult(
                        success=False,
                        field_path=field_path,
                        mapping={},
                        confidence=0.0,
                        tier=2,
                        error=str(e),
                    )
                )
                self.stats["tier2_total"] += 1

        logger.info(
            f"✅ Tier 2 complete: {self.stats['tier2_success']}/{self.stats['tier2_total']} successful"
        )
        return results

    def _execute_tier3(
        self,
        fields: List[Dict],
        previous_results: List[MappingResult],
        source_format: str,
        target_format: str,
        collection_name: str,
    ) -> List[MappingResult]:
        """Execute Tier 3: Context-dependent fields"""

        results = []
        context = self._build_context_from_results(previous_results)

        for field_info in fields:
            field_path = field_info["field"]
            logger.info(f"🔗 Processing dependent field: {field_path}")

            try:
                query = self._create_context_aware_query(field_info, context)
                chunks = self._vector_search(query, collection_name, top_k=5)

                text_mapping = self._generate_single_text_mapping(
                    field_info, chunks, context, source_format, target_format
                )

                if text_mapping:
                    json_mapping = self._text_to_json(field_path, text_mapping)

                    if json_mapping:
                        results.append(
                            MappingResult(
                                success=True,
                                field_path=field_path,
                                mapping=json_mapping,
                                confidence=0.85,
                                tier=3,
                                chunks_used=chunks[:2],
                            )
                        )
                        self.stats["tier3_success"] += 1
                    else:
                        results.append(
                            MappingResult(
                                success=False,
                                field_path=field_path,
                                mapping={},
                                confidence=0.0,
                                tier=3,
                                error="Text to JSON failed",
                            )
                        )
                else:
                    results.append(
                        MappingResult(
                            success=False,
                            field_path=field_path,
                            mapping={},
                            confidence=0.0,
                            tier=3,
                            error="Text generation failed",
                        )
                    )

                self.stats["tier3_total"] += 1

            except Exception as e:
                logger.error(f"❌ Error processing {field_path}: {e}")
                results.append(
                    MappingResult(
                        success=False,
                        field_path=field_path,
                        mapping={},
                        confidence=0.0,
                        tier=3,
                        error=str(e),
                    )
                )
                self.stats["tier3_total"] += 1

        logger.info(
            f"✅ Tier 3 complete: {self.stats['tier3_success']}/{self.stats['tier3_total']} successful"
        )
        return results

    def _process_complex_field(
        self,
        field_info: Dict,
        source_format: str,
        target_format: str,
        collection_name: str,
    ) -> MappingResult:
        """Process a single complex field"""

        field_path = field_info["field"]

        # Vector search
        query = self._create_field_query(field_info, source_format)
        chunks = self._vector_search(
            query, collection_name, top_k=self.vector_search_top_k
        )

        # Generate text mapping
        text_mapping = self._generate_single_text_mapping(
            field_info, chunks, {}, source_format, target_format
        )

        if text_mapping:
            json_mapping = self._text_to_json(field_path, text_mapping)

            if json_mapping:
                return MappingResult(
                    success=True,
                    field_path=field_path,
                    mapping=json_mapping,
                    confidence=0.80,
                    tier=2,
                    chunks_used=chunks[:3],
                )

        return MappingResult(
            success=False,
            field_path=field_path,
            mapping={},
            confidence=0.0,
            tier=2,
            error="Generation failed",
        )

    def _generate_text_mappings(
        self,
        fields: List[str],
        chunks: List[Dict],
        source_format: str,
        target_format: str,
    ) -> Optional[str]:
        """Generate simple text mappings using LLM"""

        logger.info(f"🤖 Generating text mappings for {len(fields)} fields...")

        prompt = self._build_text_prompt(fields, chunks, source_format, target_format)

        try:
            # Use generate_response instead of generate_json_response
            response = self.llama_manager.generate_response(
                prompt=prompt, max_tokens=300, temperature=0.5, timeout=300
            )

            self.stats["llm_calls"] += 1

            if response and len(response.strip()) > 10:
                logger.info(f"✅ Generated text mappings ({len(response)} chars)")
                logger.debug(f"Text mappings:\n{response[:200]}")
                return response
            else:
                logger.error("❌ Empty or short response from LLM")
                return None

        except Exception as e:
            logger.error(f"❌ Text generation failed: {e}")
            return None

    def _generate_single_text_mapping(
        self,
        field_info: Dict,
        chunks: List[Dict],
        context: Dict,
        source_format: str,
        target_format: str,
    ) -> Optional[str]:
        """Generate single field text mapping"""

        field_path = field_info["field"]
        logger.info(f"🤖 Generating text mapping for: {field_path}")

        prompt = self._build_single_text_prompt(
            field_info, chunks, context, source_format, target_format
        )

        try:
            response = self.llama_manager.generate_response(
                prompt=prompt, max_tokens=100, temperature=0.5, timeout=300
            )

            self.stats["llm_calls"] += 1

            if response and len(response.strip()) > 5:
                logger.info(f"✅ Generated text mapping")
                return response
            else:
                return None

        except Exception as e:
            logger.error(f"❌ Text generation failed: {e}")
            return None

    def _build_text_prompt(
        self,
        fields: List[str],
        chunks: List[Dict],
        source_format: str,
        target_format: str,
    ) -> str:
        """Build clearer prompt with better examples"""

        field_list = "\n".join([f"- {f.split('.')[-1]}" for f in fields[:25]])

        # Extract ACTUAL working examples from chunks
        example_mappings = []
        for chunk in chunks[:3]:
            target_node = chunk.get("entity", {}).get("target_node", {})
            refs = target_node.get("references", [])

            if refs and target_node.get("name"):
                source_path = refs[0].get("path", "")
                if source_path and "→" not in source_path:  # Avoid garbage
                    example_mappings.append(f"{target_node['name']} → {source_path}")

        examples_text = (
            "\n".join(example_mappings[:5])
            if example_mappings
            else "No examples available"
        )

        prompt = f"""You are a data mapping expert. Generate simple text mappings.

    TARGET FIELDS TO MAP (from {source_format} to {target_format}):
    {field_list}

    EXAMPLE MAPPINGS FROM SIMILAR TEMPLATES:
    {examples_text}

    YOUR TASK:
    Generate mappings in this EXACT format (one per line):
    fieldName → source.path.in.input.data

    RULES:
    1. Use the arrow symbol: →
    2. One mapping per line
    3. Map each field from the TARGET FIELDS list above
    4. The part AFTER the arrow must be a path from the SOURCE input data
    5. Keep it simple - just fieldName → sourcePath

    Generate mappings now (one per line):"""

        logger.debug(f"PROMPT:\n{prompt}")
        return prompt

    def generate_adaptive_prompt(self, field_path: str, chunks: List[Dict]) -> str:
        """
        Generate context-aware prompt based on retrieved similar mappings
        """

        # Extract patterns from retrieved chunks
        detected_patterns = {
            "segment_types": set(),
            "has_loops": False,
            "has_code_values": False,
            "has_conditions": False,
            "format_type": "unknown",
        }

        for chunk in chunks[:3]:  # Analyze top 3 chunks
            target_node = chunk.get("entity", {}).get("target_node", {})
            node_path = chunk.get("entity", {}).get("target_node_path", "")

            # Detect EDI segments
            if any(
                seg in node_path
                for seg in ["ISA-", "GS-", "ST-", "W05-", "N1-", "G62-"]
            ):
                detected_patterns["segment_types"].add("EDI")

            # Detect X12 format
            if "X12.INTERCHANGE" in node_path:
                detected_patterns["format_type"] = "X12_EDI"

            # Detect loops
            if target_node.get("loop_iterator") or target_node.get("loop_conditions"):
                detected_patterns["has_loops"] = True

            # Detect code values
            if target_node.get("code_value"):
                detected_patterns["has_code_values"] = True

            # Detect conditions
            if target_node.get("node_condition") or target_node.get("loop_conditions"):
                detected_patterns["has_conditions"] = True

        # Build adaptive instructions
        base_prompt = f"""You are an expert data mapping specialist. 

Field to map: {field_path}

Based on similar mappings, this appears to be a **{detected_patterns['format_type']}** transformation.
"""

        # Add format-specific instructions
        if "EDI" in detected_patterns["segment_types"]:
            base_prompt += """
**EDI Mapping Guidelines:**
- For ISA/GS/ST segments: Include proper segment identifiers (ISA01, GS04, etc.)
- For W05/N1/G62 segments: Map business data with qualifiers
- Use plain text for fixed values (codes, qualifiers)
- Reference source XML paths correctly

"""

        if detected_patterns["has_code_values"]:
            base_prompt += """
**Code Transformation Required:**
- Include codeValue field with Groovy/JavaScript logic
- Date formatting: Use Date().format('yyyyMMdd') or similar
- String manipulation: Use appropriate string methods
- Variable assignments: def varName = ...

"""

        if detected_patterns["has_loops"]:
            base_prompt += """
**Loop Handling Required:**
- Add loopIterator field (e.g., "it_ShipmentRefnum")
- Include loopReference with path
- Add loopConditions array with path and condition
- Example: {"path": "...Xid", "condition": "==\\"VALUE\\""}

"""

        if detected_patterns["has_conditions"]:
            base_prompt += """
**Conditional Logic Required:**
- Add nodeCondition for field-level conditions
- Use proper syntax: field == "value" or field != null

"""

        base_prompt += """
**Output Format (JSON only, no markdown):**
{
  "name": "field_name",
  "value": "variable_name_or_static_value",
  "plain": true/false,
  "codeValue": "code here if needed",
  "references": [
    {
      "jsonId": 1,
      "path": "source.path.to.field",
      "var": "variable_name",
      "text": true
    }
  ],
  "loopIterator": "iterator_name",
  "loopConditions": [
    {"path": "source.path", "condition": "== \\"value\\""}
  ]
}

**Similar Mappings for Reference:**
"""

        # Add actual examples from retrieved chunks
        for i, chunk in enumerate(chunks[:3], 1):
            target_node = chunk.get("entity", {}).get("target_node", {})
            base_prompt += (
                f"\n{i}. {chunk.get('entity', {}).get('target_node_path', 'unknown')}\n"
            )
            base_prompt += f"   Value: {target_node.get('value', 'N/A')}\n"
            if target_node.get("code_value"):
                base_prompt += (
                    f"   Code: {str(target_node.get('code_value'))[:100]}...\n"
                )
            if target_node.get("references"):
                base_prompt += f"   References: {target_node.get('references')[0].get('path', 'N/A')}\n"

        return base_prompt

    def _build_single_text_prompt(
        self,
        field_info: Dict,
        chunks: List[Dict],
        context: Dict,
        source_format: str,
        target_format: str,
    ) -> str:
        """Build prompt for single field text mapping"""

        field_path = field_info["field"]

        # USE THE NEW ADAPTIVE PROMPT INSTEAD
        return self.generate_adaptive_prompt(field_path, chunks)

    def _text_to_json(self, field_path: str, text_mappings: str) -> Optional[Dict]:
        """Convert text mapping to JSON with better error handling"""
        try:
            field_name = field_path.split(".")[-1]
            lines = text_mappings.strip().split("\n")

            for line in lines:
                line = line.strip()

                # Skip empty, examples, or malformed lines
                if not line or "→" not in line:
                    continue

                # Skip lines that look like examples/headers
                if line.startswith("#") or line.startswith("Example") or "..." in line:
                    continue

                # Parse: fieldName → source.path
                parts = line.split("→")
                if len(parts) != 2:
                    continue

                mapped_field = parts[0].strip().replace("-", "").strip()
                source_path = parts[1].strip()

                # Skip if source path looks invalid
                if not source_path or source_path.isdigit() or len(source_path) < 3:
                    continue

                # Check if this is for our field
                if (
                    mapped_field.lower() == field_name.lower()
                    or field_name.lower() in mapped_field.lower()
                ):
                    var_name = f"var{self.var_counter}"
                    self.var_counter += 1

                    return {
                        "name": field_name,
                        "value": var_name,
                        "references": [
                            {
                                "path": source_path,
                                "var": var_name,
                                "jsonId": self.var_counter - 1,
                                "text": True,
                            }
                        ],
                    }

            logger.warning(f"⚠️ No valid mapping found for {field_name}")
            return None

        except Exception as e:
            logger.error(f"❌ Text parsing failed: {e}")
            return None

    def _vector_search(
        self, query: str, collection_name: str, top_k: int = 7
    ) -> List[Dict]:
        """Perform vector similarity search using milvus_connection"""
        try:
            if not self.storage_service or not self.storage_service.is_available():
                logger.warning("⚠️ Vector search unavailable")
                return []

            # Generate embedding
            query_embedding = self.storage_service._generate_768_dim_embedding(
                query, "search_query"
            )

            if not query_embedding or len(query_embedding) != 768:
                logger.error("Invalid query embedding generated")
                return []

            # Use milvus_connection's search method directly
            from app.milvus_connection import search_similar_mappings

            results = search_similar_mappings(
                collection_name=collection_name,
                query_embedding=query_embedding,
                top_k=top_k,
                source_format=None,
                target_format=None,
                mapping_type=None,
            )

            logger.debug(f"🔍 Found {len(results)} chunks")
            return results

        except Exception as e:
            logger.error(f"❌ Vector search failed: {e}")
            return []

    def _create_group_query(self, fields: List[str], source_format: str) -> str:
        """Create query that matches storage format"""
        # Extract just the field names (last part of path)
        field_names = []
        for field_path in fields[:15]:  # Increased from 10
            parts = field_path.split(".")
            # Get meaningful parts (skip generic names like 'Gid', 'children')
            meaningful_parts = [
                p for p in parts if p not in ["Gid", "children", "root"]
            ]
            field_names.extend(meaningful_parts[-3:])  # Last 3 parts

        # Create searchable query
        query = f"SOURCE_FIELDS: {' '.join(set(field_names))}"
        return query

    def _create_field_query(self, field_info: Dict, source_format: str) -> str:
        """Create query for individual field"""
        field_path = field_info["field"]
        parts = field_path.split(".")

        # Extract meaningful parts
        meaningful_parts = [p for p in parts if p not in ["Gid", "children", "root"]]

        # Build query focusing on field names
        query_parts = ["SOURCE_FIELDS:"]
        query_parts.extend(meaningful_parts[-3:])  # Last 3 parts of path

        # Add context if available
        if field_info.get("has_loop"):
            query_parts.append("LOOP")
        if field_info.get("has_transformation"):
            query_parts.append("transformation")

        return " ".join(query_parts)

    def _create_context_aware_query(self, field_info: Dict, context: Dict) -> str:
        """Create context-aware query"""
        field_name = field_info["field"].split(".")[-1]
        return f"mapping {field_name}"

    def _extract_examples_from_chunks(self, chunks: List[Dict], limit: int = 3) -> str:
        """Extract examples from chunks"""
        if not chunks:
            return "No examples available"

        examples = []
        for i, chunk in enumerate(chunks[:limit], 1):
            entity = chunk.get("entity", {})
            name = entity.get("mapping_name", "")
            path = entity.get("target_node_path", "")

            if name and path:
                examples.append(f"{i}. {path}")

        return "\n".join(examples) if examples else "No examples"

    def _build_context_from_results(self, results: List[MappingResult]) -> Dict:
        """Build context from previous results"""
        context = {}
        for result in results:
            if result.success and result.mapping:
                context[result.field_path] = {
                    "value": result.mapping.get("value"),
                    "source_path": result.mapping.get("references", [{}])[0].get(
                        "path"
                    ),
                    "tier": result.tier,
                }
        return context

    def _build_mapping_tree(self, results: List[MappingResult]) -> Dict:
        """Build hierarchical mapping tree"""
        tree = {"name": "root", "children": []}

        for result in results:
            if result.success and result.mapping:
                tree["children"].append(result.mapping)

        return tree

    def _calculate_accuracy(self, results: List[MappingResult]) -> float:
        """Calculate accuracy percentage"""
        if not results:
            return 0.0
        successful = sum(1 for r in results if r.success)
        return round((successful / len(results)) * 100, 1)

    def _calculate_avg_confidence(self, results: List[MappingResult]) -> float:
        """Calculate average confidence"""
        if not results:
            return 0.0
        total = sum(r.confidence for r in results if r.success)
        count = sum(1 for r in results if r.success)
        return round(total / count, 2) if count > 0 else 0.0

    def _log_statistics(self, stats: Dict):
        """Log final statistics"""
        logger.info(
            f"📊 Total Fields Mapped: {stats['successful_mappings']}/{stats['total_fields']}"
        )
        logger.info(f"📊 Tier 1 Accuracy: {stats['tier1_accuracy']}%")
        logger.info(f"📊 Tier 2 Accuracy: {stats['tier2_accuracy']}%")
        logger.info(f"📊 Tier 3 Accuracy: {stats['tier3_accuracy']}%")
        logger.info(f"📊 Avg Confidence: {stats['avg_confidence']}")
        logger.info(f"📊 Total Time: {stats['total_time_seconds']}s")
        logger.info(f"📊 LLM Calls: {stats['llm_calls']}")

    def _inject_mappings_into_target_tree(
        self, target_tree: Dict, mapping_results: List[MappingResult]
    ) -> Dict:
        """
        Inject generated mappings into target tree structure

        Args:
            target_tree: Original target structure from input
            mapping_results: List of successful MappingResult objects

        Returns:
            Complete target tree with references arrays populated
        """
        if not target_tree:
            logger.warning("No target tree provided")
            return {}

        logger.info(f"💉 Injecting {len(mapping_results)} mappings into target tree...")

        # Deep copy to avoid modifying original
        import copy

        result = copy.deepcopy(target_tree)

        # Create mapping lookup by target field path
        mapping_lookup = {}
        for result_obj in mapping_results:
            if result_obj.success and result_obj.mapping:
                target_path = result_obj.field_path
                if target_path not in mapping_lookup:
                    mapping_lookup[target_path] = []
                mapping_lookup[target_path].append(result_obj.mapping)

        logger.debug(f"Created mapping lookup with {len(mapping_lookup)} paths")

        # Recursively inject mappings
        self._inject_mappings_recursive(result, "", mapping_lookup)

        logger.info(f"✅ Successfully injected {len(mapping_lookup)} field mappings")
        return result

    def _inject_mappings_recursive(
        self, node: Dict, current_path: str, mapping_lookup: Dict
    ):
        """
        Recursively traverse target tree and inject mappings

        Args:
            node: Current node in tree
            current_path: Current path from root
            mapping_lookup: Dictionary of field_path -> mapping data
        """
        if not isinstance(node, dict):
            return

        # Build current field path
        node_name = node.get("name", "")
        if current_path:
            field_path = f"{current_path}.{node_name}"
        else:
            field_path = node_name

        # Check if there are mappings for this exact field path
        if field_path in mapping_lookup:
            logger.debug(f"Found mapping for: {field_path}")

            # Get the mapping (first one if multiple)
            mapping_data = mapping_lookup[field_path][0]

            # Inject references array if it exists in the mapping
            if "references" in mapping_data:
                node["references"] = mapping_data["references"]
                logger.debug(
                    f"  ✓ Injected {len(mapping_data['references'])} references"
                )

            # Set the value if provided
            if "value" in mapping_data and not node.get("value"):
                node["value"] = mapping_data["value"]
                logger.debug(f"  ✓ Set value: {mapping_data['value']}")

            # Optionally inject other mapping properties
            for key in [
                "plain",
                "codeValue",
                "loopIterator",
                "loopReference",
                "loopConditions",
            ]:
                if key in mapping_data:
                    node[key] = mapping_data[key]

        # Recursively process children
        if "children" in node and isinstance(node["children"], list):
            for child in node["children"]:
                self._inject_mappings_recursive(child, field_path, mapping_lookup)

    def _create_mapping_from_chunk(
        self, source_field_path: str, chunks: List[Dict], source_format: str
    ) -> Optional[Dict]:
        """
        Create mapping by finding best matching chunk and adapting its references

        Args:
            source_field_path: Path from sourceTreeNode (e.g., "TransmissionBody.GLogXMLElement.SenderTransactionId")
            chunks: Retrieved similar chunks from Milvus
            source_format: Source format type

        Returns:
            Mapping dict with generated references array
        """
        try:
            # Extract source field name for matching
            source_field_name = source_field_path.split(".")[-1]

            # Find best matching chunk
            best_chunk = None
            best_score = 0.0

            for chunk in chunks[:5]:  # Check top 5 chunks
                chunk_entity = chunk.get("entity", {})
                target_node = chunk_entity.get("target_node", {})
                references = target_node.get("references", [])

                if not references:
                    continue

                # Check if any reference contains our source field
                for ref in references:
                    ref_path = ref.get("path", "")
                    ref_field_name = ref_path.split(".")[-1] if ref_path else ""

                    # Calculate similarity score
                    score = 0.0

                    # Exact match on field name
                    if ref_field_name.lower() == source_field_name.lower():
                        score = 1.0
                        best_chunk = chunk
                        break

                    # Partial match on field name
                    elif (
                        source_field_name.lower() in ref_field_name.lower()
                        or ref_field_name.lower() in source_field_name.lower()
                    ):
                        score = 0.7
                        if score > best_score:
                            best_score = score
                            best_chunk = chunk

                    # Match on path components
                    elif any(part in ref_path for part in source_field_path.split(".")):
                        score = 0.5
                        if score > best_score:
                            best_score = score
                            best_chunk = chunk

                if best_chunk and best_score >= 1.0:
                    break  # Found exact match

            if not best_chunk:
                logger.debug(f"No matching chunk found for: {source_field_name}")
                return None

            # Extract target node from best chunk
            target_node = best_chunk.get("entity", {}).get("target_node", {})

            # Generate variable name
            var_name = f"var{self.var_counter}"
            self.var_counter += 1

            # Create mapping with generated references
            mapping = {
                "name": target_node.get("name", source_field_name),
                "value": var_name,
                "references": [
                    {
                        "jsonId": self.var_counter - 1,
                        "path": source_field_path,  # ← Use the ACTUAL source path from sourceTreeNode!
                        "var": var_name,
                        "text": True,
                    }
                ],
            }

            # Copy other useful properties from chunk
            if target_node.get("plain"):
                mapping["plain"] = target_node["plain"]

            if target_node.get("code_value"):
                mapping["codeValue"] = target_node["code_value"]

            if target_node.get("loop_iterator"):
                mapping["loopIterator"] = target_node["loop_iterator"]

            if target_node.get("loop_conditions"):
                mapping["loopConditions"] = target_node["loop_conditions"]

            if target_node.get("node_condition"):
                mapping["nodeCondition"] = target_node["node_condition"]

            logger.debug(f"Created mapping: {source_field_path} → {mapping['name']}")

            return mapping

        except Exception as e:
            logger.error(f"Error creating mapping from chunk: {e}")
            return None
