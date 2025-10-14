# app/services/mapping_storage_service
import json
import hashlib
import re
from datetime import datetime
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from pymilvus import Collection
import logging

try:
    from app.config import Config
except ImportError:
    Config = None

logger = logging.getLogger(__name__)


class MappingStorageService:
    """Service for storing and searching mapping templates with 768-dim embeddings."""

    def __init__(self):
        try:
            embedding_config = (
                Config.get_embedding_config()
                if Config
                else {"model_name": "all-mpnet-base-v2", "dimension": 768}
            )
            model_name = embedding_config.get("model_name", "all-mpnet-base-v2")
            self.embedding_dimension = embedding_config.get("dimension", 768)
            self.embedding_model = SentenceTransformer(model_name)
            logger.info(f"Loaded {self.embedding_dimension}-dim model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {str(e)}")
            self.embedding_model = None
            self.embedding_dimension = 768

    def is_available(self) -> bool:
        """Check if the service is available."""
        return self.embedding_model is not None

    def is_mapped_node(self, node: dict) -> bool:
        """
        Dynamically check if a node should be stored.
        Works across different mapping file structures.
        """

        # 1. Has explicit mapping references (universal indicator)
        if node.get("references") and len(node.get("references", [])) > 0:
            return True

        # 2. Has any condition-related field (flexible matching)
        condition_keys = [
            "nodeCondition",
            "condition",
            "loopConditions",
            "loopConditionsConnective",
            "nodeConditions",
        ]
        for key in condition_keys:
            value = node.get(key)
            if value and value not in ["false", "", None, []]:
                return True

        # 3. Has any loop-related field (flexible matching)
        loop_keys = [
            "loopIterator",
            "loopReference",
            "loopStatement",
            "looper",
            "iterator",
            "loopOverRef",
        ]
        for key in loop_keys:
            if node.get(key):
                return True

        # 4. Has transformation/code (flexible matching)
        code_keys = ["codeValue", "code_value", "transformation", "script"]
        for key in code_keys:
            value = node.get(key)
            if value and isinstance(value, str) and len(value.strip()) > 0:
                return True

        # 5. Has valuable static content
        value = node.get("value", "")
        plain = node.get("plain", False)

        if value and plain:
            # Skip obvious placeholders
            if self._is_placeholder(value):
                return False
            # Store short, meaningful constants
            if len(str(value)) < 100:  # Increased from 50 for flexibility
                return True

        # 6. Has metadata/comments
        metadata_keys = ["comment", "description", "note", "metadata"]
        for key in metadata_keys:
            if node.get(key):
                return True

        # 7. Is structural parent with mapped children
        if node.get("children") and isinstance(node.get("children"), list):
            return any(self.is_mapped_node(child) for child in node.get("children", []))

        # 8. Has type indicators for arrays/objects
        node_type = node.get("type", "")
        if node_type in ["ar", "ac", "array", "object", "c"]:
            # Check if it has meaningful descendants
            if node.get("children"):
                return any(
                    self.is_mapped_node(child) for child in node.get("children", [])
                )

        return False

    def _is_placeholder(self, value: str) -> bool:
        """
        Check if value is a placeholder.
        """
        if not value:
            return True

        value_str = str(value).strip()

        # Empty or whitespace-only
        if not value_str or value_str == "":
            return True

        # Common placeholder patterns
        placeholder_patterns = [
            "#-#",  # Your specific placeholder
            "null",  # Null strings
            "undefined",  # Undefined
            "N/A",  # Not applicable
            "TBD",  # To be determined
            "TODO",  # Todo marker
        ]

        for pattern in placeholder_patterns:
            if value_str.startswith(pattern) or value_str == pattern:
                return True

        # Only spaces/special chars
        if not any(c.isalnum() for c in value_str):
            return True

        return False

    def process_mapping_data(self, mapping_data: dict) -> List[Dict[str, Any]]:
        """Process mapping data into semantic chunks for mapped nodes."""
        try:
            template_name = mapping_data.get("name", "Unknown")
            resourceId = mapping_data.get(
                "resourceId", self._generate_resourceId(template_name)
            )
            target_tree = mapping_data.get("targetTreeNode", {})
            source_tree = mapping_data.get("sourceTreeNode", {})
            local_context = mapping_data.get("localContext", {})

            logger.info(
                f"Processing mapping: {template_name} (resourceId: {resourceId})"
            )

            # Extract mapped nodes and create chunks
            chunks = []
            index = self._extract_mapped_nodes(
                target_tree,
                template_name,
                resourceId,
                source_tree,
                local_context,
                chunks,
                path="root",
                index=0,
            )

            if not chunks:
                logger.warning(f"No mapped nodes found in template: {template_name}")
            else:
                logger.info(
                    f"Created {len(chunks)} chunks for template: {template_name}"
                )
            return chunks
        except Exception as e:
            logger.error(f"Error processing mapping data: {str(e)}", exc_info=True)
            return []

    def _extract_mapped_nodes(
        self,
        node: dict,
        template_name: str,
        resourceId: str,
        source_tree: dict,
        local_context: dict,
        chunks: List[Dict],
        path: str = "root",
        index: int = 0,
    ) -> int:
        """Recursively extract mapped nodes and create chunks."""
        if not self.is_mapped_node(node):
            return index

        # Create chunk for the current mapped node
        node_name = node.get("name", "")
        current_path = f"{path}.{node_name}" if path != "root" else node_name

        try:
            chunk = {
                "chunk_id": f"{resourceId}_{current_path}_{index}",
                "mapping_name": template_name,
                "resource_id": resourceId,
                "source_input_type": "EDI",
                "target_input_type": "JSON",
                "target_node_path": current_path,
                "target_node": {
                    "name": node_name,
                    "value": node.get("value", ""),
                    "type": node.get("type", ""),
                    "code_value": node.get("codeValue", node.get("code_value", None)),
                    "node_condition": node.get("nodeCondition", None),
                    "loop_iterator": node.get(
                        "loopIterator",
                        node.get("looper", {}).get("loopStatement", None),
                    ),
                    "loop_conditions": self._extract_loop_conditions(node),
                    "loop_conditions_connective": node.get(
                        "loopConditionsConnective", None
                    ),
                    "references": node.get("references", []),
                },
                "functions": self._extract_functions(node),
                "prolog": self._extract_prolog(node, local_context),
                "vector_embedding": None,
            }

            chunk["vector_embedding"] = self._generate_768_dim_embedding(
                self._create_chunk_text(chunk), "mapped_node"
            )
            chunks.append(chunk)
            logger.debug(f"Created chunk for: {current_path}")
            index += 1
        except Exception as e:
            logger.error(
                f"Error creating chunk for {current_path}: {str(e)}", exc_info=True
            )

        # Process children
        for child in node.get("children", []):
            index = self._extract_mapped_nodes(
                child,
                template_name,
                resourceId,
                source_tree,
                local_context,
                chunks,
                current_path,
                index,
            )

        return index

    def _extract_loop_conditions(self, node: dict) -> List[str]:
        """Extract loop conditions as strings."""
        loop_conditions = node.get("loopConditions", [])
        if not loop_conditions:
            return []

        conditions = []
        for cond in loop_conditions:
            if isinstance(cond, dict):
                # Extract path and condition
                path = cond.get("path", "")
                condition = cond.get("condition", "")
                conditions.append(f"{path} {condition}")
            elif isinstance(cond, str):
                conditions.append(cond)
        return conditions

    # def _create_chunk_text(self, chunk: dict) -> str:
    #     """Create text representation of a chunk for embedding."""
    #     node = chunk["target_node"]
    #     text_parts = [
    #         f"MAPPING: {chunk['mapping_name']}",
    #         f"PATH: {chunk['target_node_path']}",
    #         f"NAME: {node['name']}",
    #     ]

    #     if node.get("value"):
    #         text_parts.append(f"VALUE: {node['value']}")
    #     if node.get("code_value"):
    #         code_snippet = (
    #             str(node["code_value"])[:100] + "..."
    #             if len(str(node["code_value"])) > 100
    #             else str(node["code_value"])
    #         )
    #         text_parts.append(f"CODE: {code_snippet}")
    #     if node.get("node_condition"):
    #         text_parts.append(f"CONDITION: {node['node_condition']}")
    #     if node.get("loop_iterator"):
    #         text_parts.append(f"LOOP: {node['loop_iterator']}")
    #     if node.get("loop_conditions"):
    #         # Ensure all conditions are strings
    #         conditions_str = "; ".join([str(c) for c in node["loop_conditions"]])
    #         text_parts.append(f"LOOP_CONDITIONS: {conditions_str}")
    #     if node.get("references"):
    #         refs = []
    #         for ref in node["references"]:
    #             source_path = ref.get("path", ref.get("source_path", ""))
    #             refs.append(f"{source_path} → {chunk['target_node_path']}")
    #         text_parts.append(f"REFERENCES: {'; '.join(refs)}")
    #     if chunk.get("functions"):
    #         funcs = [f"{f['name']}: {f['short_value']}" for f in chunk["functions"]]
    #         text_parts.append(f"FUNCTIONS: {'; '.join(funcs)}")
    #     if chunk.get("prolog"):
    #         prolog_snippet = (
    #             str(chunk["prolog"])[:100] + "..."
    #             if len(str(chunk["prolog"])) > 100
    #             else str(chunk["prolog"])
    #         )
    #         text_parts.append(f"PROLOG: {prolog_snippet}")

    #     return "\n".join(text_parts)

    def _create_chunk_text(self, chunk: dict) -> str:
        """Create searchable text representation including both source and target context."""
        node = chunk["target_node"]
        text_parts = []

        # 1. MOST IMPORTANT: Source field paths from references
        if node.get("references"):
            source_paths = []
            for ref in node["references"]:
                source_path = ref.get("path", "")
                if source_path:
                    # Extract field names for better matching
                    source_fields = [p for p in source_path.split(".") if p]
                    source_paths.extend(source_fields)

            if source_paths:
                text_parts.append(f"SOURCE_FIELDS: {' '.join(source_paths)}")

        # 2. Target field information
        target_fields = chunk["target_node_path"].split(".")
        text_parts.append(f"TARGET_FIELDS: {' '.join(target_fields)}")
        text_parts.append(f"TARGET_NAME: {node['name']}")

        # 3. Value and transformation context
        if node.get("value"):
            text_parts.append(f"VALUE: {node['value']}")

        if node.get("code_value"):
            # Extract keywords from code for better matching
            code_str = str(node["code_value"])
            # Extract variable names and function calls
            vars_found = re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", code_str)
            if vars_found:
                text_parts.append(f"CODE_VARS: {' '.join(set(vars_found[:10]))}")

        if node.get("loop_iterator"):
            text_parts.append(f"LOOP: {node['loop_iterator']}")

        if node.get("node_condition"):
            text_parts.append(f"CONDITION: {node['node_condition']}")

        # 4. Mapping context (less important, at the end)
        text_parts.append(f"MAPPING_TYPE: XML to EDI transformation")

        return "\n".join(text_parts)

    def _extract_functions(self, node: dict) -> List[Dict]:
        """Extract function details from node."""
        functions = []
        code_value = node.get("codeValue", node.get("code_value", ""))
        if code_value and isinstance(code_value, str) and "def " in code_value:
            # Extract function name if possible
            func_name_match = re.search(r"def\s+(\w+)\s*\(", code_value)
            func_name = (
                func_name_match.group(1) if func_name_match else "custom_function"
            )

            functions.append(
                {
                    "name": func_name,
                    "signature": f"def {func_name}()",
                    "value": code_value,
                    "short_value": code_value[:50] + "..."
                    if len(code_value) > 50
                    else code_value,
                }
            )
        return functions

    def _extract_prolog(self, node: dict, local_context: dict) -> str:
        """Extract prolog script."""
        # Check local context first
        if local_context:
            output_prolog = local_context.get("outputProlog", {})
            if output_prolog and isinstance(output_prolog, dict):
                prolog_value = output_prolog.get("value", "")
                if prolog_value:
                    return prolog_value

        # Check node prolog
        return node.get("prolog", "")

    def _generate_768_dim_embedding(self, text: str, pattern_type: str) -> List[float]:
        """Generate 768-dim embedding."""
        try:
            if self.embedding_model and text.strip():
                enhanced_text = (
                    f"DATA MAPPING: {text}" if pattern_type != "search_query" else text
                )
                embedding = self.embedding_model.encode(
                    enhanced_text, convert_to_tensor=False
                ).tolist()
                return (
                    embedding
                    if len(embedding) == self.embedding_dimension
                    else [0.0] * self.embedding_dimension
                )
            logger.warning("Embedding model unavailable or empty text")
            return [0.0] * self.embedding_dimension
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            return [0.0] * self.embedding_dimension

    def _generate_resourceId(self, template_name: str) -> str:
        """Generate unique resource ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{template_name.replace(' ', '_')}_{timestamp}"

    def _generate_chunk_id(self, base_id: str) -> str:
        """Generate unique chunk ID."""
        return hashlib.md5(
            f"{base_id}_{datetime.now().timestamp()}".encode()
        ).hexdigest()[:16]

    # In MappingStorageService
    def search_mappings(self, query: str, top_k: int = 10) -> List[Dict]:
        """Search for similar mapping patterns"""
        try:
            from app.milvus_connection import search_similar_mappings

            query_embedding = self._generate_768_dim_embedding(query, "search_query")

            if not query_embedding or len(query_embedding) != 768:
                logger.error("Invalid query embedding")
                return []

            return search_similar_mappings(
                collection_name="OTMToJT", query_embedding=query_embedding, top_k=top_k
            )

        except Exception as e:
            logger.error(f"Search failed: {e}", exc_info=True)
            return []
