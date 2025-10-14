

# app/services/field_mapper.py - COMPLETE FILE WITH TRANSFORMATION HINTS
import json
import logging
import re
import torch
from typing import Dict, Any, Optional, List, Set, Tuple

logger = logging.getLogger(__name__)


class FocusedFieldMapper:
    """Handles dynamic field mapping through vector database-driven pattern learning with transformation hints"""

    def __init__(self, llama_manager):
        self.llama_manager = llama_manager

        # Transformation code templates matching your vector DB storage
        self.transformation_templates = {
            "numeric_conversion": lambda var: f"var{var} = MapperUtility.convertToDouble(var{var})",
            "hyphen_removal": lambda var: f'if(var{var} != null) {{ var{var} = var{var}.replace("-","") }}',
            "uppercase_conversion": lambda var: f"if(var{var} != null) {{ var{var} = var{var}.toUpperCase() }}",
            "lowercase_conversion": lambda var: f"if(var{var} != null) {{ var{var} = var{var}.toLowerCase() }}",
            "direct": lambda var: None,  # No transformation
            "carrier_code_mapping": lambda var: None,  # Complex - handled separately
            "array_element": lambda var: None,  # Structural
            "conditional": lambda var: None,  # Complex
        }

    def generate_focused_field_mappings(
        self, input_data: Dict[str, Any], template: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate mappings using hybrid base + MULTIPLE LLM enhancement calls with transformation hints"""

        logger.info(
            "Starting Hybrid Base + Multi-Pass LLM Enhancement with Transformation Hints..."
        )

        # Clear GPU before starting
        self._clear_gpu_cache()

        # Step 1: Run hybrid mapping first (baseline)
        input_analysis = self._analyze_input_structure_comprehensive(input_data)
        template_patterns = self._extract_comprehensive_template_patterns(template)

        logger.info("Step 1: Generating hybrid base mappings...")
        hybrid_mappings = self._vector_database_driven_comprehensive_mapping(
            template_patterns, input_data, input_analysis
        )

        hybrid_tree = self._convert_mappings_to_target_tree_node(
            hybrid_mappings, input_analysis
        )
        hybrid_mapped_paths = self._get_mapped_paths_from_tree(hybrid_tree)

        logger.info(f"Hybrid base: {len(hybrid_mapped_paths)} fields mapped")

        # Clear GPU after hybrid processing
        self._clear_gpu_cache()

        # Step 2: Extract concrete mapping examples WITH transformation metadata
        logger.info(
            "Step 2: Extracting concrete mapping examples with transformations..."
        )
        concrete_examples = (
            self._extract_concrete_mapping_examples_with_transformations(template)
        )

        if not concrete_examples or len(concrete_examples) < 2:
            logger.warning("No concrete examples found, using hybrid only")
            return {
                "targetTreeNode": hybrid_tree,
                "generation_method": "hybrid_only_no_examples",
            }

        # Step 3: Find ALL unmapped fields
        all_input_paths = input_analysis["field_paths"]
        unmapped_paths = [p for p in all_input_paths if p not in hybrid_mapped_paths]

        if len(unmapped_paths) == 0:
            logger.info("All fields mapped by hybrid")
            return {
                "targetTreeNode": hybrid_tree,
                "generation_method": "hybrid_complete",
            }

        logger.info(f"Step 3: Found {len(unmapped_paths)} unmapped fields")

        # Step 4: Multiple LLM calls for better coverage with transformation hints
        logger.info(
            "Step 4: Starting multi-pass LLM enhancement with transformation hints..."
        )

        combined_tree = hybrid_tree
        current_mapped_paths = hybrid_mapped_paths.copy()
        total_llm_added = 0

        # Make up to 2 LLM calls, each handling ~6 fields
        max_llm_calls = 2
        fields_per_call = 6

        for call_num in range(1, max_llm_calls + 1):
            # CRITICAL: Clear CUDA memory before EACH LLM call
            logger.info(f"\n--- LLM Call {call_num}/{max_llm_calls} ---")
            logger.info("Clearing GPU memory before LLM call...")
            self._clear_gpu_cache()

            # Get next batch of unmapped fields
            remaining_unmapped = [
                p for p in unmapped_paths if p not in current_mapped_paths
            ]

            if len(remaining_unmapped) == 0:
                logger.info(f"All fields mapped after {call_num-1} LLM calls")
                break

            logger.info(
                f"Processing {min(len(remaining_unmapped), fields_per_call)} fields"
            )

            # Create prompt with transformation hints for this batch
            enhancement_prompt = self._create_enhancement_prompt_with_hints(
                concrete_examples,
                remaining_unmapped[: fields_per_call * 2],  # Give it more options
                input_data,
                current_mapped_paths,
            )

            if not enhancement_prompt:
                logger.warning(f"Could not create prompt for call {call_num}")
                break

            # Call LLM with shorter timeout per call
            llm_response = None
            if self.llama_manager and self.llama_manager.is_model_loaded():
                try:
                    logger.info(f"Calling LLM for batch {call_num}...")
                    llm_response = self.llama_manager.generate_json_response(
                        enhancement_prompt,
                        max_tokens=600,  # Fixed per call
                        temperature=0.2,  # Low for consistency
                        timeout=240,  # 4 min per call
                        top_p=0.85,
                        top_k=30,
                    )

                    # CRITICAL: Clear GPU immediately after LLM call
                    logger.info("Clearing GPU memory after LLM call...")
                    self._clear_gpu_cache()

                except Exception as e:
                    logger.error(f"LLM call {call_num} failed: {e}")
                    # Clear GPU even on error
                    self._clear_gpu_cache()
                    break

            if not llm_response or not llm_response.strip():
                logger.warning(f"LLM call {call_num} returned no response")
                continue  # Try next call instead of breaking

            # Parse simple text output with transformation hints
            try:
                parsed_mappings = self._parse_llm_simple_output_with_hints(
                    llm_response, len(current_mapped_paths) + 1
                )

                if not parsed_mappings or len(parsed_mappings) == 0:
                    logger.warning(f"LLM call {call_num} returned no valid mappings")
                    continue  # Try next call

                logger.info(
                    f"LLM call {call_num} generated {len(parsed_mappings)} mappings"
                )

                # Convert to tree structure
                llm_tree = {"name": "root", "children": parsed_mappings}

                # Merge with combined tree
                prev_count = len(current_mapped_paths)
                combined_tree = self._merge_llm_enhancements_with_hybrid(
                    combined_tree,
                    llm_tree,
                    current_mapped_paths,
                )

                # Update tracking
                current_mapped_paths = self._get_mapped_paths_from_tree(combined_tree)
                added_this_call = len(current_mapped_paths) - prev_count
                total_llm_added += added_this_call

                logger.info(
                    f"Added {added_this_call} new mappings from call {call_num}"
                )
                logger.info(
                    f"Total mapped so far: {len(current_mapped_paths)}/{len(all_input_paths)}"
                )

            except Exception as e:
                logger.error(f"Error parsing LLM output for call {call_num}: {e}")
                continue  # Try next call

            # Clear GPU after processing this batch
            self._clear_gpu_cache()

        # Clear GPU one final time after all calls
        logger.info("Clearing GPU memory after all LLM calls...")
        self._clear_gpu_cache()

        # Final result
        final_mapped_paths = self._get_mapped_paths_from_tree(combined_tree)
        total_fields = len(all_input_paths)
        coverage_pct = (
            (len(final_mapped_paths) / total_fields * 100) if total_fields > 0 else 0
        )

        logger.info(f"\n=== FINAL RESULTS ===")
        logger.info(
            f"Total: {len(final_mapped_paths)}/{total_fields} fields ({coverage_pct:.1f}%)"
        )
        logger.info(f"  - Hybrid: {len(hybrid_mapped_paths)} mappings")
        logger.info(
            f"  - LLM added: {total_llm_added} mappings ({max_llm_calls} calls)"
        )
        logger.info(f"  - Coverage: {coverage_pct:.1f}%")

        return {
            "targetTreeNode": combined_tree,
            "generation_method": "hybrid_plus_multi_pass_llm_with_transformation_hints",
            "hybrid_contributed": len(hybrid_mapped_paths),
            "llm_contributed": total_llm_added,
            "total_mapped": len(final_mapped_paths),
            "coverage_percentage": round(coverage_pct, 2),
            "llm_calls_made": call_num if call_num <= max_llm_calls else max_llm_calls,
        }

    def _extract_concrete_mapping_examples_with_transformations(
        self, template: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Extract examples from BOTH targetTreeNode AND vector DB metadata"""

        all_examples = []

        # METHOD 1: Extract from reconstructed template's targetTreeNode
        target_tree = template.get("targetTreeNode", {})

        if target_tree and "children" in target_tree:
            children = target_tree.get("children", [])

            logger.info(f"Method 1: Extracting from {len(children)} template children")

            for child in children:
                if isinstance(child, dict) and self._is_complete_mapping(child):
                    source_path = ""
                    if child.get("references"):
                        source_path = child["references"][0].get("path", "")

                    transform_type = self._classify_transformation_from_mapping(child)

                    all_examples.append(
                        {
                            "name": child.get("name", ""),
                            "source_path": source_path,
                            "transformation_type": transform_type,
                            "codeValue": child.get("codeValue", ""),
                            "full_mapping": child,
                        }
                    )

        # METHOD 2: Extract from vector DB micro-chunk metadata
        micro_chunks = template.get("source_micro_chunks_data", [])

        if micro_chunks:
            logger.info(
                f"Method 2: Extracting from {len(micro_chunks)} micro-chunks metadata"
            )

            for chunk in micro_chunks:
                try:
                    # CRITICAL FIX: Access Hit object attributes correctly
                    metadata = None

                    # Try different ways to access metadata
                    if hasattr(chunk, "entity"):
                        # It's a search result Hit object
                        if hasattr(chunk.entity, "metadata"):
                            metadata = chunk.entity.metadata
                        elif hasattr(chunk.entity, "get"):
                            metadata = chunk.entity.get("metadata")
                    elif isinstance(chunk, dict):
                        metadata = chunk.get("metadata", {})
                    else:
                        continue

                    if not metadata:
                        continue

                    # FIXED: Handle JSON-encoded metadata
                    if isinstance(metadata, str):
                        try:
                            metadata = json.loads(metadata)
                        except:
                            continue

                    # Extract mapping_details from metadata
                    mapping_details = (
                        metadata.get("mapping_details", {})
                        if isinstance(metadata, dict)
                        else {}
                    )

                    # Get direct mappings from metadata
                    direct_mappings = mapping_details.get("direct_mappings", [])
                    for mapping in direct_mappings[:3]:  # Take first 3 from each chunk
                        if isinstance(mapping, dict):
                            target_field = mapping.get("target_field", "")
                            if target_field.startswith("root."):
                                target_field = target_field[5:]
                            if "." in target_field:
                                target_field = target_field.split(".")[-1]

                            all_examples.append(
                                {
                                    "name": target_field,
                                    "source_path": mapping.get("source_path", ""),
                                    "transformation_type": "direct",
                                    "codeValue": "",
                                    "from_metadata": True,
                                }
                            )

                    # Get transformation details from metadata
                    transformation_details = mapping_details.get(
                        "transformation_details", {}
                    )
                    transformations = transformation_details.get("transformations", [])

                    for transform in transformations[:3]:
                        if isinstance(transform, dict):
                            target_field = transform.get("target_field", "")
                            if target_field.startswith("root."):
                                target_field = target_field[5:]
                            if "." in target_field:
                                target_field = target_field.split(".")[-1]

                            all_examples.append(
                                {
                                    "name": target_field,
                                    "source_path": transform.get("source_path", ""),
                                    "transformation_type": transform.get(
                                        "transformation_type", "custom_transformation"
                                    ),
                                    "codeValue": transform.get(
                                        "transformation_code", ""
                                    ),
                                    "from_metadata": True,
                                }
                            )

                except Exception as e:
                    logger.debug(f"Could not extract from chunk metadata: {e}")
                    continue

        if len(all_examples) == 0:
            logger.warning("No examples extracted from either method")
            return []

        # Select diverse examples
        diverse_examples = self._select_diverse_transformation_examples(
            all_examples, max_count=8
        )

        logger.info(
            f"Extracted {len(diverse_examples)} transformation-aware examples (from targetTree + metadata)"
        )
        return diverse_examples

    def _classify_transformation_from_mapping(self, mapping: Dict[str, Any]) -> str:
        """Classify what type of transformation a mapping uses"""

        code_value = mapping.get("codeValue", "")

        if not code_value:
            return "direct"

        code_lower = code_value.lower()

        if "converttodouble" in code_lower or "todouble" in code_lower:
            return "numeric_conversion"
        elif "replace" in code_lower and '"-"' in code_value:
            return "hyphen_removal"
        elif "touppercase" in code_lower:
            return "uppercase_conversion"
        elif "tolowercase" in code_lower:
            return "lowercase_conversion"
        else:
            return "custom_transformation"

    def _select_diverse_transformation_examples(
        self, all_examples: List[Dict[str, Any]], max_count: int = 8
    ) -> List[Dict[str, Any]]:
        """Select diverse examples covering different transformation types"""

        if len(all_examples) <= max_count:
            return all_examples

        diverse_examples = []
        types_seen = set()

        # Priority order: get one of each type first
        priority_types = [
            "numeric_conversion",
            "hyphen_removal",
            "direct",
            "uppercase_conversion",
            "lowercase_conversion",
            "custom_transformation",
        ]

        # First pass: get one example of each priority type
        for priority_type in priority_types:
            for example in all_examples:
                if (
                    example["transformation_type"] == priority_type
                    and example not in diverse_examples
                ):
                    diverse_examples.append(example)
                    types_seen.add(priority_type)
                    break

            if len(diverse_examples) >= max_count:
                break

        # Second pass: fill remaining slots with any good examples
        for example in all_examples:
            if example not in diverse_examples:
                diverse_examples.append(example)

                if len(diverse_examples) >= max_count:
                    break

        logger.info(
            f"Selected {len(diverse_examples)} diverse examples covering types: {types_seen}"
        )
        return diverse_examples

    def _create_enhancement_prompt_with_hints(
        self,
        concrete_examples: List[Dict[str, Any]],
        unmapped_paths: List[str],
        input_data: Dict[str, Any],
        already_mapped: Set[str],
    ) -> str:
        """Create ULTRA-SIMPLE prompt to prevent garbage output"""

        priority_unmapped = self._select_priority_fields(unmapped_paths, input_data)[
            :3
        ]  # ONLY 3 fields

        if len(priority_unmapped) == 0:
            return ""

        next_var = len(already_mapped) + 1

        # Build VERY simple expected output showing EXACTLY what we want
        expected_lines = []
        for i, field in enumerate(priority_unmapped):
            var_num = next_var + i
            field_name = field["path"].split(".")[-1]

            # Determine transformation type from field value
            value_sample = field.get("value_sample", "")
            if "-" in str(value_sample):
                transform = "hyphen_removal"
            elif value_sample and str(value_sample).replace(".", "").isdigit():
                transform = "numeric_conversion"
            else:
                transform = "direct"

            expected_lines.append(f'{field["path"]} → {field_name} [{transform}]')

        # ULTRA-SIMPLE PROMPT
        prompt = f"""[INST] Map 3 fields using this format:

    EXAMPLE FORMAT:
    root.origin → originLocation [direct]
    root.refNum → awbNumber [hyphen_removal]
    root.weight → totalWeight [numeric_conversion]

    YOUR TASK - Map these 3 fields:
    1. {priority_unmapped[0]["path"]} (value: "{priority_unmapped[0]["value_sample"]}")
    2. {priority_unmapped[1]["path"] if len(priority_unmapped) > 1 else "N/A"}
    3. {priority_unmapped[2]["path"] if len(priority_unmapped) > 2 else "N/A"}

    TRANSFORMATION TYPES:
    [direct] [hyphen_removal] [numeric_conversion]

    OUTPUT (3 lines only):
    {expected_lines[0]}
    {expected_lines[1] if len(expected_lines) > 1 else ''}
    {expected_lines[2] if len(expected_lines) > 2 else ''}

    Generate now: [/INST]

    """

        return prompt

    def _parse_llm_simple_output_with_hints(
        self, llm_response: str, start_var: int
    ) -> List[Dict[str, Any]]:
        """Parse simple LLM text output with transformation hints into proper JSON structure"""

        mappings = []
        lines = llm_response.strip().split("\n")
        var_counter = start_var

        logger.info(f"Parsing LLM output starting at var{start_var}...")

        for line in lines:
            # Skip empty lines
            if not line.strip():
                continue

            # Must contain the arrow
            if "→" not in line:
                continue

            try:
                # Parse format: "root.destination → destinationLocation [direct]"
                parts = line.split("→")
                if len(parts) != 2:
                    logger.debug(f"Skipping malformed line: {line}")
                    continue

                source_path = parts[0].strip()
                right_side = parts[1].strip()

                # Extract target name and transformation hint
                if "[" in right_side and "]" in right_side:
                    target_name = right_side.split("[")[0].strip()
                    transform_hint = right_side.split("[")[1].split("]")[0].strip()
                else:
                    target_name = right_side
                    transform_hint = "direct"

                # Validate transformation type
                if transform_hint not in self.transformation_templates:
                    logger.warning(
                        f"Unknown transformation type: {transform_hint}, defaulting to 'direct'"
                    )
                    transform_hint = "direct"

                # Build proper JSON mapping structure
                mapping = {
                    "name": target_name,
                    "value": f"var{var_counter}",
                    "references": [
                        {
                            "jsonId": var_counter,
                            "path": source_path,
                            "var": f"var{var_counter}",
                            "text": True,
                        }
                    ],
                }

                # Look up and add transformation code from templates
                transform_code = self._get_transformation_code(
                    transform_hint, var_counter
                )
                if transform_code:
                    mapping["codeValue"] = transform_code

                mappings.append(mapping)
                var_counter += 1

                logger.debug(
                    f"✓ Parsed: {source_path} → {target_name} [{transform_hint}]"
                )

            except Exception as e:
                logger.warning(f"Failed to parse line: '{line}' - {e}")
                continue

        logger.info(f"Successfully parsed {len(mappings)} mappings from LLM output")
        return mappings

    def _get_transformation_code(
        self, transform_type: str, var_num: int
    ) -> Optional[str]:
        """Get actual transformation code from transformation type using templates"""

        if transform_type in self.transformation_templates:
            template_func = self.transformation_templates[transform_type]
            code = template_func(var_num)
            if code:
                logger.debug(f"Applied {transform_type} transformation to var{var_num}")
            return code

        logger.debug(f"No transformation code for type: {transform_type}")
        return None

    def _select_priority_fields(
        self, unmapped_paths: List[str], input_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Select high-priority unmapped fields"""

        priority_fields = []

        for path in unmapped_paths:
            value = self._get_value_at_path(path, input_data)
            category = self._categorize_field_from_path(path)
            priority = self._calculate_mapping_priority(path, value, category)

            if priority > 30:  # Only include meaningful fields
                priority_fields.append(
                    {
                        "path": path,
                        "category": category,
                        "value_sample": str(value)[:50] if value else "",
                        "priority": priority,
                    }
                )

        # Sort by priority
        priority_fields.sort(key=lambda x: x["priority"], reverse=True)

        return priority_fields

    def _get_mapped_paths_from_tree(self, tree: Dict[str, Any]) -> Set[str]:
        """Extract all mapped paths from target tree"""

        mapped_paths = set()

        def extract_paths(node):
            if isinstance(node, dict):
                # Extract from references
                for ref in node.get("references", []):
                    if isinstance(ref, dict) and "path" in ref:
                        path = ref["path"]
                        if isinstance(path, str):
                            mapped_paths.add(path)

                # Recurse children
                for child in node.get("children", []):
                    extract_paths(child)

        extract_paths(tree)
        return mapped_paths

    def _merge_llm_enhancements_with_hybrid(
        self,
        hybrid_tree: Dict[str, Any],
        llm_tree: Dict[str, Any],
        hybrid_mapped_paths: Set[str],
    ) -> Dict[str, Any]:
        """Merge LLM's additional mappings with hybrid base"""

        combined_tree = json.loads(json.dumps(hybrid_tree))

        # Get highest var number from hybrid
        max_var = self._find_highest_var_in_tree(combined_tree)
        next_var = max_var + 1

        logger.info(f"Merging LLM enhancements starting at var{next_var}")

        # Add LLM children that don't duplicate hybrid mappings
        llm_children = llm_tree.get("children", [])
        added_count = 0

        for child in llm_children:
            if not isinstance(child, dict):
                continue

            # Get path from this child
            child_refs = child.get("references", [])
            if not child_refs:
                continue

            child_path = child_refs[0].get("path", "")

            # Skip if already mapped by hybrid
            if child_path in hybrid_mapped_paths:
                logger.debug(f"Skipping duplicate path: {child_path}")
                continue

            # Add with updated var numbers
            new_child = json.loads(json.dumps(child))

            # Update var numbers
            for ref in new_child.get("references", []):
                ref["jsonId"] = next_var
                ref["var"] = f"var{next_var}"
                ref["source"] = "llm_transformation_hint"

            new_child["value"] = f"var{next_var}"

            # Update codeValue if present
            if "codeValue" in new_child:
                new_child["codeValue"] = re.sub(
                    r"var\d+", f"var{next_var}", new_child["codeValue"]
                )

            combined_tree["children"].append(new_child)
            hybrid_mapped_paths.add(child_path)
            next_var += 1
            added_count += 1

        logger.info(f"Added {added_count} LLM enhancements to hybrid base")

        return combined_tree

    def _find_highest_var_in_tree(self, tree: Dict[str, Any]) -> int:
        """Find highest var number in tree"""
        max_var = 0

        def scan(node):
            nonlocal max_var
            if isinstance(node, dict):
                # Check value
                value = node.get("value", "")
                if isinstance(value, str) and value.startswith("var"):
                    try:
                        num = int(value.replace("var", ""))
                        max_var = max(max_var, num)
                    except:
                        pass

                # Check references
                for ref in node.get("references", []):
                    if isinstance(ref, dict):
                        var_val = ref.get("var", "")
                        if isinstance(var_val, str) and var_val.startswith("var"):
                            try:
                                num = int(var_val.replace("var", ""))
                                max_var = max(max_var, num)
                            except:
                                pass

                # Recurse
                for child in node.get("children", []):
                    scan(child)

        scan(tree)
        return max_var

    def _is_complete_mapping(self, mapping: Dict[str, Any]) -> bool:
        """Check if mapping has all required fields"""
        if not isinstance(mapping, dict):
            return False

        # Must have name, value, and references
        if (
            "name" not in mapping
            or "value" not in mapping
            or "references" not in mapping
        ):
            return False

        # References must be non-empty
        refs = mapping.get("references", [])
        if not isinstance(refs, list) or len(refs) == 0:
            return False

        # Check first reference has path
        first_ref = refs[0]
        if not isinstance(first_ref, dict) or "path" not in first_ref:
            return False

        path = first_ref["path"]
        if not path or not isinstance(path, str) or len(path) < 5:
            return False

        return True

    def _clear_gpu_cache(self):
        """Clear GPU cache to prevent OOM"""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.debug("GPU cache cleared")
        except Exception as e:
            logger.debug(f"Could not clear GPU cache: {e}")

    # ==================== EXISTING HELPER METHODS (UNCHANGED FROM YOUR ORIGINAL) ====================

    def _vector_database_driven_comprehensive_mapping(
        self,
        template_patterns: Dict[str, Any],
        input_data: Dict[str, Any],
        input_analysis: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Vector database mapping (unchanged)"""

        mappings = {
            "direct_fields": {},
            "transformed_fields": {},
            "numeric_fields": {},
            "array_fields": {},
            "nested_fields": {},
        }

        field_examples = template_patterns.get("field_mapping_examples", [])
        mapping_candidates = input_analysis["mapping_candidates"]

        for candidate in mapping_candidates[:30]:
            source_path = candidate["path"]
            field_category = candidate["category"]
            field_type = candidate["type"]

            best_pattern = self._find_best_vector_pattern(candidate, field_examples)

            if best_pattern:
                target_field = self._generate_target_field_name(candidate, best_pattern)

                if best_pattern["mapping_type"] == "transformed":
                    transform_type = best_pattern.get("transformation_type", "custom")
                    mappings["transformed_fields"][target_field] = (
                        f"{source_path}|{transform_type}"
                    )
                elif (
                    field_type in ["numeric", "numeric_string"]
                    and field_category == "measurement"
                ):
                    mappings["numeric_fields"][target_field] = (
                        f"{source_path}|to_number"
                    )
                else:
                    mappings["direct_fields"][target_field] = source_path
            else:
                target_field = self._generate_intelligent_target_field_name(candidate)

                if field_category == "identifier":
                    if "refnum" in source_path.lower() or "awb" in source_path.lower():
                        mappings["transformed_fields"][target_field] = (
                            f"{source_path}|remove_hyphens"
                        )
                    else:
                        mappings["direct_fields"][target_field] = source_path
                elif field_category == "measurement":
                    mappings["numeric_fields"][target_field] = (
                        f"{source_path}|to_number"
                    )
                elif field_category in ["location", "tracking", "routing", "temporal"]:
                    mappings["direct_fields"][target_field] = source_path
                elif candidate["priority"] > 50:
                    mappings["direct_fields"][target_field] = source_path

        return mappings

    def _convert_mappings_to_target_tree_node(
        self, mappings: Dict[str, Any], input_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Convert mapping dict to targetTreeNode structure"""

        children = []
        var_counter = 1

        for target_name, source_path in mappings.get("direct_fields", {}).items():
            children.append(
                {
                    "name": target_name,
                    "value": f"var{var_counter}",
                    "references": [
                        {
                            "jsonId": var_counter,
                            "path": source_path,
                            "var": f"var{var_counter}",
                            "text": True,
                        }
                    ],
                }
            )
            var_counter += 1

        for target_name, source_mapping in mappings.get("numeric_fields", {}).items():
            source_path = (
                source_mapping.split("|")[0]
                if "|" in source_mapping
                else source_mapping
            )
            children.append(
                {
                    "name": target_name,
                    "value": f"var{var_counter}",
                    "references": [
                        {
                            "jsonId": var_counter,
                            "path": source_path,
                            "var": f"var{var_counter}",
                            "text": True,
                        }
                    ],
                    "codeValue": f"var{var_counter} = MapperUtility.convertToDouble(var{var_counter})",
                }
            )
            var_counter += 1

        for target_name, source_mapping in mappings.get(
            "transformed_fields", {}
        ).items():
            parts = source_mapping.split("|")
            source_path = parts[0]
            transform = parts[1] if len(parts) > 1 else None

            node = {
                "name": target_name,
                "value": f"var{var_counter}",
                "references": [
                    {
                        "jsonId": var_counter,
                        "path": source_path,
                        "var": f"var{var_counter}",
                        "text": True,
                    }
                ],
            }

            if transform == "remove_hyphens":
                node["codeValue"] = (
                    f'if(var{var_counter} != null) {{ var{var_counter} = var{var_counter}.replace("-","") }}'
                )
            elif transform == "uppercase":
                node["codeValue"] = (
                    f"if(var{var_counter} != null) {{ var{var_counter} = var{var_counter}.toUpperCase() }}"
                )

            children.append(node)
            var_counter += 1

        return {"name": "root", "children": children}

    def _extract_comprehensive_template_patterns(
        self, template: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract patterns from template"""

        patterns = {
            "field_mapping_examples": [],
            "transformation_patterns": [],
            "array_processing_patterns": [],
        }

        target_tree = template.get("targetTreeNode", {})
        if target_tree:
            self._extract_field_mapping_examples_comprehensive(
                target_tree, patterns["field_mapping_examples"]
            )

        return patterns

    def _extract_field_mapping_examples_comprehensive(
        self, node: Dict[str, Any], examples: List[Dict[str, Any]], path: str = ""
    ):
        """Extract field mapping examples"""

        if isinstance(node, dict):
            current_path = (
                f"{path}.{node.get('name', '')}" if path else node.get("name", "")
            )

            references = node.get("references", [])
            if references:
                for ref in references:
                    source_path = ref.get("path", "")
                    var_name = ref.get("var", "")
                    if source_path and var_name:
                        example = {
                            "target_field": current_path,
                            "source_path": source_path,
                            "variable": var_name,
                            "mapping_type": "direct",
                            "field_category": self._categorize_field_from_path(
                                source_path
                            ),
                        }

                        code_value = node.get("codeValue", "")
                        if code_value:
                            example["mapping_type"] = "transformed"
                            example["transformation"] = code_value

                        examples.append(example)

            for child in node.get("children", []):
                self._extract_field_mapping_examples_comprehensive(
                    child, examples, current_path
                )

    def _analyze_input_structure_comprehensive(
        self, input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze input structure"""

        analysis = {
            "total_fields": 0,
            "field_paths": [],
            "mapping_candidates": [],
        }

        all_paths = self._extract_all_field_paths_comprehensive(input_data)
        analysis["field_paths"] = all_paths
        analysis["total_fields"] = len(all_paths)

        for path in all_paths:
            value = self._get_value_at_path(path, input_data)
            if value is not None:
                field_type = self._determine_field_type_comprehensive(value, path)
                category = self._categorize_field_from_path(path)

                if self._is_high_value_field(path, value):
                    analysis["mapping_candidates"].append(
                        {
                            "path": path,
                            "type": field_type,
                            "category": category,
                            "value_sample": str(value)[:50] if value else "",
                            "priority": self._calculate_mapping_priority(
                                path, value, category
                            ),
                        }
                    )

        analysis["mapping_candidates"].sort(key=lambda x: x["priority"], reverse=True)

        return analysis

    def _extract_all_field_paths_comprehensive(
        self, obj: Any, prefix: str = "root"
    ) -> List[str]:
        """Extract all field paths"""
        paths = []

        def extract_recursive(data, current_path, depth=0):
            if depth > 10:
                return

            if isinstance(data, dict):
                for key, value in data.items():
                    new_path = f"{current_path}.{key}"
                    paths.append(new_path)
                    extract_recursive(value, new_path, depth + 1)
            elif isinstance(data, list) and data:
                paths.append(f"{current_path}[0]")
                extract_recursive(data[0], f"{current_path}[0]", depth + 1)

        extract_recursive(obj, prefix)
        return paths

    def _get_value_at_path(self, path: str, input_data: Dict[str, Any]) -> Any:
        """Get value at path"""
        try:
            parts = path.replace("root.", "").split(".")
            current = input_data
            for part in parts:
                if "[0]" in part:
                    array_name = part.split("[")[0]
                    if (
                        array_name in current
                        and isinstance(current[array_name], list)
                        and current[array_name]
                    ):
                        current = current[array_name][0]
                    else:
                        return None
                elif part in current:
                    current = current[part]
                else:
                    return None
            return current
        except:
            return None

    def _categorize_field_from_path(self, path: str) -> str:
        """Categorize field from path"""
        path_lower = path.lower()

        if any(kw in path_lower for kw in ["refnum", "awb", "tracking"]):
            return "identifier"
        elif any(kw in path_lower for kw in ["origin", "destination"]):
            return "location"
        elif any(kw in path_lower for kw in ["weight", "pieces", "volume"]):
            return "measurement"
        elif any(kw in path_lower for kw in ["event", "status", "milestone"]):
            return "tracking"
        elif any(kw in path_lower for kw in ["route", "flight", "segment"]):
            return "routing"
        elif any(kw in path_lower for kw in ["date", "time"]):
            return "temporal"
        else:
            return "general"

    def _determine_field_type_comprehensive(self, value: Any, path: str) -> str:
        """Determine field type"""
        if isinstance(value, (int, float)):
            return "numeric"
        elif isinstance(value, str):
            if self._is_numeric_string(value):
                return "numeric_string"
            else:
                return "text"
        elif isinstance(value, list):
            return "array"
        elif isinstance(value, dict):
            return "object"
        else:
            return "unknown"

    def _is_high_value_field(self, path: str, value: Any) -> bool:
        """Check if field is high value"""
        path_lower = path.lower()
        high_priority_keywords = [
            "refnum",
            "awb",
            "tracking",
            "origin",
            "destination",
            "weight",
            "pieces",
            "volume",
            "eventcode",
            "status",
        ]

        if any(kw in path_lower for kw in high_priority_keywords):
            return True

        if value and str(value).strip():
            return True

        return False

    def _calculate_mapping_priority(self, path: str, value: Any, category: str) -> int:
        """Calculate mapping priority"""
        score = 0
        path_lower = path.lower()

        category_scores = {
            "identifier": 100,
            "location": 90,
            "measurement": 80,
            "tracking": 70,
            "routing": 60,
            "temporal": 50,
            "general": 30,
        }
        score += category_scores.get(category, 20)

        if "refnum" in path_lower or "awb" in path_lower:
            score += 50
        if any(kw in path_lower for kw in ["origin", "destination"]):
            score += 40
        if any(kw in path_lower for kw in ["weight", "pieces"]):
            score += 30

        nesting_level = len(path.split(".")) - 1
        score -= min(nesting_level * 5, 30)

        if value and len(str(value).strip()) > 0:
            score += 10

        return max(score, 0)

    def _find_best_vector_pattern(
        self, candidate: Dict[str, Any], field_examples: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Find best matching pattern from vector examples"""

        candidate_path = candidate["path"]
        candidate_category = candidate["category"]

        best_match = None
        best_score = 0

        for example in field_examples:
            score = 0

            example_path = example.get("source_path", "")
            example_category = example.get("field_category", "")

            if candidate_category == example_category:
                score += 50

            candidate_field = candidate_path.split(".")[-1].lower()
            example_field = example_path.split(".")[-1].lower()

            if candidate_field == example_field:
                score += 100
            elif candidate_field in example_field or example_field in candidate_field:
                score += 30

            if score > best_score:
                best_score = score
                best_match = example

        return best_match if best_score > 40 else None

    def _generate_target_field_name(
        self, candidate: Dict[str, Any], pattern: Dict[str, Any]
    ) -> str:
        """Generate target field name from pattern"""
        return pattern.get("target_field", candidate["path"].split(".")[-1])

    def _generate_intelligent_target_field_name(self, candidate: Dict[str, Any]) -> str:
        """Generate intelligent target field name"""
        path_parts = candidate["path"].split(".")
        field_name = path_parts[-1]
        clean_name = re.sub(r"[^a-zA-Z0-9]", "_", field_name)

        if len(path_parts) >= 3:
            context = path_parts[-2]
            if context not in clean_name:
                clean_name = f"{context}_{clean_name}"

        return clean_name

    def _is_numeric_string(self, value: str) -> bool:
        """Check if string is numeric"""
        try:
            float(str(value))
            return True
        except:
            return False

    def extract_search_features(self, input_data: Dict[str, Any]) -> str:
        """Extract search features"""
        all_paths = self._extract_all_field_paths_comprehensive(input_data)
        return f"total_fields:{len(all_paths)}"

    def assess_input_quality(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess input quality"""
        all_paths = self._extract_all_field_paths_comprehensive(input_data)
        return {
            "total_fields": len(all_paths),
            "complexity_score": min(len(all_paths) / 50, 1.0),
            "pattern_learning_potential": len(all_paths) > 10,
        }
