"""
Field Analyzer - Extract and categorize fields for Smart Hybrid mapping
Handles EDI, XML, and JSON input formats dynamically
"""

import re
import logging
from typing import List, Dict, Any, Set, Tuple
from collections import defaultdict
from dataclasses import dataclass, field as dataclass_field

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class Field:
    """Represents a field to be mapped"""

    path: str  # Hierarchical path (e.g., "root.shipment.shipper.name")
    name: str  # Field name
    type: str  # Field type (string, int, array, object, etc.)
    value: Any = None  # Sample value if available
    parent_path: str = ""  # Parent hierarchy
    depth: int = 0  # Nesting level
    is_array: bool = False  # Is this field in an array context?
    array_context: str = ""  # Name of parent array if applicable

    # Categorization flags
    category: str = "unknown"  # static, simple, complex, dependent
    complexity_score: float = 0.0  # 0-1 scale

    # Metadata
    has_transformation: bool = False
    has_condition: bool = False
    has_loop: bool = False
    requires_context: bool = False
    related_fields: List[str] = dataclass_field(default_factory=list)
    group_id: str = ""  # Assigned group identifier


class FieldAnalyzer:
    """Extract and categorize fields from unmapped input files"""

    def __init__(self):
        self.fields: List[Field] = []
        self.field_map: Dict[str, Field] = {}  # path -> Field
        self.groups: Dict[str, List[Field]] = defaultdict(list)

        # Patterns for complexity detection
        self.transformation_keywords = {
            "date",
            "format",
            "convert",
            "parse",
            "calculate",
            "sum",
            "aggregate",
            "join",
            "split",
            "substring",
        }

        self.static_patterns = {
            "id",
            "type",
            "code",
            "version",
            "status",
            "constant",
            "fixed",
            "standard",
        }

    def analyze_fields(
        self, input_data: dict, input_format: str = "json"
    ) -> Dict[str, Any]:
        """
        Main entry point - analyze input file and categorize fields
        Compatible with both raw data and tree node structures

        Args:
            input_data: Parsed input (can be raw JSON or tree node structure)
            input_format: "json", "edi", "xml", "JSON", "XML"

        Returns:
            Analysis results with categorized fields and groups
        """
        logger.info(
            f"🔍 Analyzing {input_format.upper()} input for field extraction..."
        )

        # Reset state for new analysis
        self.fields = []
        self.field_map = {}
        self.groups = defaultdict(list)

        # Check if this is a tree node structure (sourceTreeNode format)
        if self._is_tree_node_structure(input_data):
            logger.info("📋 Detected tree node structure (sourceTreeNode format)")
            self._extract_from_tree_nodes(input_data)
        else:
            # Standard extraction for raw JSON/XML
            logger.info("📋 Detected standard data structure")
            self._extract_fields(input_data, input_format.lower())

        logger.info(f"📊 Extracted {len(self.fields)} fields")

        # Step 2: Categorize each field
        self._categorize_fields()

        # Step 3: Group related fields
        self._group_fields()

        # Step 4: Generate analysis report
        analysis = self._generate_analysis_report()

        return analysis

    def _is_tree_node_structure(self, data: dict) -> bool:
        """
        Check if input is a tree node structure (has 'name' and 'children')
        This is the format from sourceTreeNode
        """
        if not isinstance(data, dict):
            return False

        # Tree node structure has 'name' and optionally 'children', 'value', 'plain'
        return "name" in data and (
            "children" in data or "value" in data or "plain" in data
        )

    def _extract_from_tree_nodes(
        self,
        node: dict,
        path: str = "",
        parent_path: str = "",
        depth: int = 0,
        array_context: str = "",
    ):
        """
        Extract fields from tree node structure (sourceTreeNode format)

        Args:
            node: Tree node with structure: {name, value?, plain?, children?}
            path: Current hierarchical path
            parent_path: Parent node path
            depth: Nesting depth
            array_context: Parent array context if applicable
        """
        if not isinstance(node, dict):
            return

        node_name = node.get("name", "")
        if not node_name:
            return

        # Build current path
        current_path = f"{path}.{node_name}" if path else node_name

        # Check if this is a leaf node (has 'value' or 'plain' flag)
        is_leaf = "plain" in node and node.get("plain") is True
        node_value = node.get("value")

        if is_leaf or (
            node_value is not None and not isinstance(node_value, (dict, list))
        ):
            # This is a mappable field!
            field_obj = Field(
                path=current_path,
                name=node_name,
                type=self._infer_type(node_value)
                if node_value is not None
                else "string",
                value=node_value,
                parent_path=parent_path or path,
                depth=depth,
                is_array=bool(array_context),
                array_context=array_context,
            )

            self.fields.append(field_obj)
            self.field_map[current_path] = field_obj

            logger.debug(f"Extracted field: {current_path} (value: {node_value})")

        # Recursively process children
        children = node.get("children", [])
        if isinstance(children, list):
            for child in children:
                if isinstance(child, dict):
                    self._extract_from_tree_nodes(
                        child, current_path, path or "root", depth + 1, array_context
                    )

    def _extract_fields(
        self,
        data: Any,
        input_format: str,
        path: str = "root",
        parent_path: str = "",
        depth: int = 0,
        array_context: str = "",
    ):
        """Recursively extract fields from nested structure"""

        if isinstance(data, dict):
            for key, value in data.items():
                current_path = f"{path}.{key}" if path != "root" else key

                # Create field object
                field_obj = Field(
                    path=current_path,
                    name=key,
                    type=self._infer_type(value),
                    value=value if not isinstance(value, (dict, list)) else None,
                    parent_path=parent_path or path,
                    depth=depth,
                    is_array=bool(array_context),
                    array_context=array_context,
                )

                self.fields.append(field_obj)
                self.field_map[current_path] = field_obj

                # Recurse into nested structures
                if isinstance(value, dict):
                    self._extract_fields(
                        value,
                        input_format,
                        current_path,
                        path,
                        depth + 1,
                        array_context,
                    )
                elif isinstance(value, list) and len(value) > 0:
                    # Mark as array and process first item as template
                    field_obj.is_array = True
                    if isinstance(value[0], dict):
                        self._extract_fields(
                            value[0], input_format, current_path, path, depth + 1, key
                        )

        elif isinstance(data, list):
            # Handle top-level arrays
            if len(data) > 0 and isinstance(data[0], dict):
                self._extract_fields(
                    data[0], input_format, path, parent_path, depth, "root_array"
                )

    def _infer_type(self, value: Any) -> str:
        """Infer field type from value"""
        if value is None:
            return "unknown"
        elif isinstance(value, bool):
            return "boolean"
        elif isinstance(value, int):
            return "integer"
        elif isinstance(value, float):
            return "float"
        elif isinstance(value, str):
            return "string"
        elif isinstance(value, list):
            return "array"
        elif isinstance(value, dict):
            return "object"
        else:
            return "unknown"

    def _categorize_fields(self):
        """Categorize each field as static, simple, complex, or dependent"""

        for field in self.fields:
            # Skip container objects/arrays
            if field.type in ["object", "array"]:
                field.category = "container"
                continue

            # Calculate complexity score
            complexity_score = 0.0

            # Factor 1: Static value indicators (REDUCES complexity)
            field_lower = field.name.lower()
            if any(pattern in field_lower for pattern in self.static_patterns):
                complexity_score -= 0.3

            # Factor 2: Transformation keywords (INCREASES complexity)
            if any(kw in field_lower for kw in self.transformation_keywords):
                complexity_score += 0.4
                field.has_transformation = True

            # Factor 3: Array context (INCREASES complexity)
            if field.is_array or field.array_context:
                complexity_score += 0.3
                field.has_loop = True

            # Factor 4: Depth complexity (deeper = more complex)
            complexity_score += min(field.depth * 0.1, 0.3)

            # Factor 5: Value type complexity
            if field.type in ["float", "boolean"]:
                complexity_score += 0.2  # Often requires calculation

            # Factor 6: Sample value analysis
            if field.value is not None:
                if self._looks_like_constant(field.value):
                    complexity_score -= 0.2
                if self._looks_like_calculated(field.value):
                    complexity_score += 0.3

            # Normalize score to 0-1
            field.complexity_score = max(0.0, min(1.0, complexity_score))

            # Assign category based on score
            if field.complexity_score <= 0.2:
                field.category = "static"
            elif field.complexity_score <= 0.5:
                field.category = "simple"
            elif field.complexity_score <= 0.75:
                field.category = "complex"
            else:
                field.category = "dependent"
                field.requires_context = True

            logger.debug(
                f"Field {field.path}: {field.category} "
                f"(score: {field.complexity_score:.2f})"
            )

    def _looks_like_constant(self, value: Any) -> bool:
        """Check if value looks like a constant"""
        if not isinstance(value, str):
            return False

        # Short uppercase strings often constants (EDI codes, etc.)
        if len(value) <= 10 and value.isupper():
            return True

        # Fixed patterns
        if re.match(r"^[A-Z0-9_-]+$", value):
            return True

        return False

    def _looks_like_calculated(self, value: Any) -> bool:
        """Check if value looks calculated/derived"""
        if not isinstance(value, str):
            return isinstance(value, (int, float))  # Numeric often calculated

        # Contains calculation indicators
        calc_patterns = ["sum", "total", "count", "avg", "max", "min"]
        return any(pattern in value.lower() for pattern in calc_patterns)

    def _group_fields(self):
        """Group related fields for batch processing"""

        # Strategy 1: Group by parent hierarchy (most important)
        hierarchy_groups = defaultdict(list)
        for field in self.fields:
            if field.category != "container":
                # Group by parent path up to depth 2
                parts = field.parent_path.split(".")[:3]
                group_key = ".".join(parts)
                hierarchy_groups[group_key].append(field)

        # Strategy 2: Group by category within hierarchies
        for hierarchy_key, hierarchy_fields in hierarchy_groups.items():
            category_buckets = defaultdict(list)

            for field in hierarchy_fields:
                category_buckets[field.category].append(field)

            # Create final groups
            for category, fields in category_buckets.items():
                if len(fields) > 0:
                    group_id = f"{hierarchy_key}_{category}"

                    # Split large groups (max 25 fields per group for LLM)
                    if len(fields) > 25:
                        for i in range(0, len(fields), 25):
                            chunk = fields[i : i + 25]
                            chunk_group_id = f"{group_id}_part{i//25 + 1}"
                            for field in chunk:
                                field.group_id = chunk_group_id
                            self.groups[chunk_group_id] = chunk
                    else:
                        for field in fields:
                            field.group_id = group_id
                        self.groups[group_id] = fields

        # logger.info(f"📦 Created {len(self.groups)} field groups")
        if self.groups:
            sample_groups = list(self.groups.items())[:3]
            for group_id, fields in sample_groups:
                if len(fields) > 0:
                    category = fields[0].category
                    logger.debug(f"  {group_id}: {len(fields)} {category} fields")
            
            if len(self.groups) > 3:
                logger.debug(f"  ... and {len(self.groups) - 3} more groups")

        # Log group statistics
        # Log only summary of groups (first 3 groups)
        logged_count = 0
        for group_id, fields in self.groups.items():
            if len(fields) > 0:
                category = fields[0].category
                if logged_count < 3:
                    logger.debug(
                        f"Group {group_id}: {len(fields)} {category} fields"
                    )
                    logged_count += 1
                elif logged_count == 3:
                    remaining = len(self.groups) - 3
                    if remaining > 0:
                        logger.debug(f"... and {remaining} more groups")
                    logged_count += 1
                    break

    def _generate_analysis_report(self) -> Dict[str, Any]:
        """Generate comprehensive analysis report"""

        # Count by category
        category_counts = defaultdict(int)
        for field in self.fields:
            if field.category != "container":
                category_counts[field.category] += 1

        # Tier assignments
        tier_assignments = {
            "tier1_static_simple": [],  # Batch processing
            "tier2_complex": [],  # Individual processing
            "tier3_dependent": [],  # Context-dependent
        }

        for group_id, fields in self.groups.items():
            if len(fields) == 0:
                continue

            category = fields[0].category

            if category in ["static", "simple"]:
                tier_assignments["tier1_static_simple"].append(
                    {
                        "group_id": group_id,
                        "field_count": len(fields),
                        "fields": [f.path for f in fields],
                        "category": category,
                    }
                )
            elif category == "complex":
                tier_assignments["tier2_complex"].extend(
                    [
                        {
                            "field": f.path,
                            "complexity_score": f.complexity_score,
                            "has_loop": f.has_loop,
                            "has_transformation": f.has_transformation,
                        }
                        for f in fields
                    ]
                )
            else:  # dependent
                tier_assignments["tier3_dependent"].extend(
                    [
                        {
                            "field": f.path,
                            "requires_context": f.requires_context,
                            "related_fields": f.related_fields,
                        }
                        for f in fields
                    ]
                )

        report = {
            "summary": {
                "total_fields": len(
                    [f for f in self.fields if f.category != "container"]
                ),
                "total_groups": len(self.groups),
                "by_category": dict(category_counts),
            },
            "tier_assignments": tier_assignments,
            "field_details": [
                {
                    "path": f.path,
                    "category": f.category,
                    "complexity_score": f.complexity_score,
                    "group_id": f.group_id,
                    "is_array": f.is_array,
                    "depth": f.depth,
                }
                for f in self.fields
                if f.category != "container"
            ],
            "groups": {
                group_id: {
                    "size": len(fields),
                    "category": fields[0].category if fields else "unknown",
                    "fields": [f.path for f in fields],
                }
                for group_id, fields in self.groups.items()
            },
        }

        # Log summary
        logger.info("📊 FIELD ANALYSIS SUMMARY:")
        logger.info(f"  Total mappable fields: {report['summary']['total_fields']}")
        logger.info(f"  Static fields: {category_counts['static']}")
        logger.info(f"  Simple fields: {category_counts['simple']}")
        logger.info(f"  Complex fields: {category_counts['complex']}")
        logger.info(f"  Dependent fields: {category_counts['dependent']}")
        logger.info(f"  Tier 1 groups: {len(tier_assignments['tier1_static_simple'])}")
        logger.info(f"  Tier 2 fields: {len(tier_assignments['tier2_complex'])}")
        logger.info(f"  Tier 3 fields: {len(tier_assignments['tier3_dependent'])}")

        return report

    def get_fields_by_tier(self, tier: int) -> List[Field]:
        """Get all fields assigned to a specific tier"""
        if tier == 1:
            categories = ["static", "simple"]
        elif tier == 2:
            categories = ["complex"]
        else:  # tier 3
            categories = ["dependent"]

        return [
            f
            for f in self.fields
            if f.category in categories and f.category != "container"
        ]

    def get_group_fields(self, group_id: str) -> List[Field]:
        """Get all fields in a specific group"""
        return self.groups.get(group_id, [])
