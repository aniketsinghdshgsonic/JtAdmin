import os
import logging
from typing import List, Dict, Any
from pymilvus import (
    connections,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    utility,
)
from datetime import datetime
import json

logger = logging.getLogger(__name__)

# Collection configurations for 768-dim embeddings
COLLECTION_CONFIGS = {
    "OTMToJT": {
        "description": "Air cargo field mappings with 768-dim embeddings",
        "dimension": 768,
        "index_params": {
            "metric_type": "COSINE",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 2048},
        },
    },
    "JTToFedex": {
        "description": "JT to FedEx field mappings with 768-dim",
        "dimension": 768,
        "index_params": {
            "metric_type": "COSINE",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 2048},
        },
    },
}


def connect_to_milvus():
    """Connect to Milvus using environment variables."""
    try:
        milvus_host = os.getenv("MILVUS_HOST", "localhost")
        milvus_port = int(os.getenv("MILVUS_PORT", "19530"))
        connections.connect(alias="default", host=milvus_host, port=milvus_port)
        logger.info(f"Connected to Milvus at {milvus_host}:{milvus_port}")
        return True
    except Exception as e:
        logger.error(f"Failed to connect to Milvus: {str(e)}")
        return False


def disconnect_from_milvus():
    """
    Disconnect from Milvus
    """
    try:
        connections.disconnect("default")
        logger.info("Disconnected from Milvus")
    except Exception as e:
        logger.error(f"Error disconnecting from Milvus: {str(e)}")


def get_all_collections_stats():
    """
    Get enhanced statistics for both mapping collections

    Returns:
        Dictionary with stats for both collections
    """
    collection_names = ["OTMToJT", "JTToFedex"]
    all_stats = {}

    for collection_name in collection_names:
        stats = get_collection_stats(collection_name)
        all_stats[collection_name] = stats

    return all_stats


def get_collection_stats(collection_name):
    """
    Get enhanced statistics about a Milvus collection

    Args:
        collection_name (str): Name of the collection

    Returns:
        Dictionary with collection statistics
    """
    try:
        if not utility.has_collection(collection_name):
            return {"error": f"Collection '{collection_name}' does not exist"}

        collection = Collection(collection_name)
        collection.load()

        stats = {
            "name": collection_name,
            "num_entities": collection.num_entities,
            "schema_fields": [field.name for field in collection.schema.fields],
            "enhanced_schema": "awb_pattern"
            in [field.name for field in collection.schema.fields],
            "indexes": [index.field_name for index in collection.indexes],
        }

        # Try to get additional stats if collection has data
        if collection.num_entities > 0:
            try:
                # Sample query to get some metadata
                sample_results = collection.query(
                    expr="id >= 0",
                    limit=10,
                    output_fields=[
                        "template_name",
                        "mapping_type",
                        "chunk_type",
                        "complexity_score",
                    ],
                )

                if sample_results:
                    templates = list(
                        set(
                            r.get("template_name", "")
                            for r in sample_results
                            if r.get("template_name")
                        )
                    )
                    mapping_types = list(
                        set(
                            r.get("mapping_type", "")
                            for r in sample_results
                            if r.get("mapping_type")
                        )
                    )
                    chunk_types = list(
                        set(
                            r.get("chunk_type", "")
                            for r in sample_results
                            if r.get("chunk_type")
                        )
                    )

                    stats.update(
                        {
                            "sample_templates": templates[:5],
                            "sample_mapping_types": mapping_types[:5],
                            "sample_chunk_types": chunk_types[:5],
                            "avg_complexity": sum(
                                r.get("complexity_score", 0) for r in sample_results
                            )
                            / len(sample_results),
                        }
                    )

            except Exception as sample_error:
                logger.debug(
                    f"Could not get sample stats for {collection_name}: {sample_error}"
                )

        return stats

    except Exception as e:
        logger.error(
            f"Failed to get collection stats for '{collection_name}': {str(e)}"
        )
        return {"error": str(e)}


def create_collection(collection_name: str, dimension: int = 768) -> Collection:
    """Create a Milvus collection for 768-dim embeddings."""
    try:
        if utility.has_collection(collection_name):
            collection = Collection(collection_name)
            embedding_field = next(
                (
                    field
                    for field in collection.schema.fields
                    if field.name == "embedding"
                ),
                None,
            )
            if embedding_field and embedding_field.params.get("dim") != dimension:
                logger.warning(
                    f"Dimension mismatch in '{collection_name}'. Recreating..."
                )
                utility.drop_collection(collection_name)
            else:
                logger.info(f"Using existing collection '{collection_name}'")
                return collection

        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dimension),
            FieldSchema(name="metadata", dtype=DataType.JSON),
            FieldSchema(name="template_name", dtype=DataType.VARCHAR, max_length=255),
            FieldSchema(name="resourceId", dtype=DataType.VARCHAR, max_length=255),
            FieldSchema(name="version", dtype=DataType.INT32),
            FieldSchema(name="is_active", dtype=DataType.BOOL),
            FieldSchema(
                name="pattern_category", dtype=DataType.VARCHAR, max_length=100
            ),
            FieldSchema(name="chunk_type", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="pattern_type", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(
                name="business_domain_tags", dtype=DataType.VARCHAR, max_length=1000
            ),
            FieldSchema(name="source_fields", dtype=DataType.VARCHAR, max_length=2000),
            FieldSchema(name="target_fields", dtype=DataType.VARCHAR, max_length=2000),
            FieldSchema(name="complexity_score", dtype=DataType.FLOAT),
            FieldSchema(name="has_transformations", dtype=DataType.BOOL),
            FieldSchema(name="has_arrays", dtype=DataType.BOOL),
            FieldSchema(name="has_carrier_code_mapping", dtype=DataType.BOOL),
            FieldSchema(name="created_at", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="source_format", dtype=DataType.VARCHAR, max_length=255),
            FieldSchema(name="target_format", dtype=DataType.VARCHAR, max_length=255),
            FieldSchema(name="mapping_type", dtype=DataType.VARCHAR, max_length=50),
        ]

        schema = CollectionSchema(fields, f"Mapping collection for {collection_name}")
        collection = Collection(collection_name, schema)
        config = COLLECTION_CONFIGS.get(collection_name, COLLECTION_CONFIGS["OTMToJT"])
        collection.create_index(
            field_name="embedding", index_params=config["index_params"]
        )
        logger.info(
            f"Created collection '{collection_name}' with dimension {dimension}"
        )
        return collection
    except Exception as e:
        logger.error(f"Failed to create collection '{collection_name}': {str(e)}")
        return None


def _split_large_metadata_into_chunks(
    metadata: Dict[str, Any], max_bytes: int = 60000
) -> List[Dict[str, Any]]:
    """Split large metadata into chunks, prioritizing mapping_details."""
    try:
        current_size = len(json.dumps(metadata).encode("utf-8"))
        if current_size <= max_bytes:
            return [metadata]

        logger.info(f"Splitting metadata ({current_size} bytes) into chunks...")
        chunks = []
        chunk_counter = 0
        base_metadata = metadata.copy()
        mapping_details = base_metadata.pop("mapping_details", None)

        if mapping_details:
            direct_mappings = mapping_details.get("direct_mappings", [])
            nested_mappings = mapping_details.get("nested_mappings", [])
            constant_mappings = mapping_details.get("constant_mappings", [])
            total_mappings = (
                len(direct_mappings) + len(nested_mappings) + len(constant_mappings)
            )

            if total_mappings > 0:
                mappings_per_chunk = max(
                    1,
                    (max_bytes - len(json.dumps(base_metadata).encode("utf-8")))
                    // 2000,
                )
                i = 0
                while i < max(
                    len(direct_mappings), len(nested_mappings), len(constant_mappings)
                ):
                    chunk_metadata = base_metadata.copy()
                    chunk_metadata["mapping_details"] = {
                        "direct_mappings": direct_mappings[i : i + mappings_per_chunk],
                        "nested_mappings": nested_mappings[i : i + mappings_per_chunk],
                        "constant_mappings": constant_mappings[
                            i : i + mappings_per_chunk
                        ],
                    }
                    chunk_metadata["direct_mapping_count"] = len(
                        chunk_metadata["mapping_details"]["direct_mappings"]
                    )
                    chunk_metadata["nested_mapping_count"] = len(
                        chunk_metadata["mapping_details"]["nested_mappings"]
                    )
                    chunk_metadata["constant_mapping_count"] = len(
                        chunk_metadata["mapping_details"]["constant_mappings"]
                    )
                    chunk_metadata["_is_split"] = True
                    chunk_metadata["_chunk_index"] = chunk_counter
                    chunk_metadata["_total_chunks"] = (
                        total_mappings + mappings_per_chunk - 1
                    ) // mappings_per_chunk
                    chunks.append(chunk_metadata)
                    chunk_counter += 1
                    i += mappings_per_chunk
                logger.info(
                    f"Split {total_mappings} mappings into {len(chunks)} chunks"
                )
                return chunks

        logger.warning("No mapping_details to split, returning original metadata")
        return [metadata]
    except Exception as e:
        logger.error(f"Failed to split metadata: {str(e)}")
        return [metadata]


def insert_mapping_embeddings(
    collection_name: str,
    texts: List[str],
    embeddings: List[List[float]],
    metadata_list: List[Dict[str, Any]],
    source_format: str,
    target_format: str,
    mapping_type: str,
    template_name: str,
    resourceId: str,
    version: int = 1,
) -> Any:
    """Insert mapping embeddings with metadata splitting."""
    try:
        collection = Collection(collection_name)
        collection.load()
        all_enhanced_data = []

        for i, (text, embedding, metadata) in enumerate(
            zip(texts, embeddings, metadata_list)
        ):
            metadata_chunks = _split_large_metadata_into_chunks(metadata)
            for chunk_idx, metadata_chunk in enumerate(metadata_chunks):
                enhanced_item = {
                    "text": text[:65534],
                    "embedding": embedding,
                    "metadata": metadata_chunk,
                    "template_name": template_name[:254],
                    "resourceId": resourceId[:254],
                    "version": version,
                    "is_active": True,
                    "pattern_category": metadata_chunk.get(
                        "pattern_category", "unknown"
                    )[:99],
                    "chunk_type": metadata_chunk.get("chunk_type", mapping_type)[:99],
                    "pattern_type": metadata_chunk.get("pattern_type", "unknown")[:99],
                    "business_domain_tags": " ".join(
                        metadata_chunk.get("business_domain_tags", [])
                    )[:999],
                    "source_fields": " ".join(metadata_chunk.get("source_fields", []))[
                        :1999
                    ],
                    "target_fields": " ".join(metadata_chunk.get("target_fields", []))[
                        :1999
                    ],
                    "complexity_score": float(
                        metadata_chunk.get("complexity_score", 0.5)
                    ),
                    "has_transformations": bool(
                        metadata_chunk.get("has_transformations", False)
                    ),
                    "has_arrays": bool(metadata_chunk.get("has_arrays", False)),
                    "has_carrier_code_mapping": bool(
                        metadata_chunk.get("has_carrier_code_mapping", False)
                    ),
                    "created_at": metadata_chunk.get(
                        "created_at", datetime.now().isoformat()
                    )[:49],
                    "source_format": source_format[:254],
                    "target_format": target_format[:254],
                    "mapping_type": mapping_type[:49],
                }
                all_enhanced_data.append(enhanced_item)
                if len(metadata_chunks) > 1:
                    logger.info(f"Split item {i} into {len(metadata_chunks)} chunks")

        field_names = [
            "text",
            "embedding",
            "metadata",
            "template_name",
            "resourceId",
            "version",
            "is_active",
            "pattern_category",
            "chunk_type",
            "pattern_type",
            "business_domain_tags",
            "source_fields",
            "target_fields",
            "complexity_score",
            "has_transformations",
            "has_arrays",
            "has_carrier_code_mapping",
            "created_at",
            "source_format",
            "target_format",
            "mapping_type",
        ]
        insert_data = [
            [item[field] for item in all_enhanced_data] for field in field_names
        ]
        insert_result = collection.insert(insert_data)
        collection.flush()
        logger.info(
            f"Inserted {len(all_enhanced_data)} embeddings into '{collection_name}'"
        )
        return insert_result
    except Exception as e:
        logger.error(f"Failed to insert embeddings into '{collection_name}': {str(e)}")
        return None


def search_similar_mappings(
    collection_name: str,
    query_embedding: List[float],
    top_k: int = 10,
    source_format: str = None,
    target_format: str = None,
    mapping_type: str = None,
) -> List[Dict[str, Any]]:
    """
    Search for similar mappings using COSINE similarity.
    Returns formatted results compatible with orchestrator.
    """
    try:
        collection = Collection(collection_name)
        collection.load()
        
        # Specify only the fields we need (those that exist in schema)
        output_fields = [
            "template_name",
            "resourceId", 
            "text",           # Contains target_node_path
            "metadata",       # Contains target_node data
            "chunk_type",
            "pattern_type",
            "source_fields",
            "target_fields",
            "complexity_score",
            "created_at"
        ]
        
        # Search parameters for COSINE similarity
        search_params = {
            "metric_type": "COSINE", 
            "params": {"nprobe": 32}  # Increased for better recall
        }
        
        # Build filter expression
        filter_conditions = []
        if source_format:
            filter_conditions.append(f'source_format == "{source_format}"')
        if target_format:
            filter_conditions.append(f'target_format == "{target_format}"')
        if mapping_type:
            filter_conditions.append(f'mapping_type == "{mapping_type}"')
        
        expr = " && ".join(filter_conditions) if filter_conditions else None
        
        # Execute search
        results = collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            expr=expr,
            output_fields=output_fields,
        )
        
        if not results or len(results[0]) == 0:
            logger.warning(f"No results found in '{collection_name}'")
            return []
        
        logger.info(
            f"Found {len(results[0])} similar mappings in '{collection_name}' "
            f"(avg distance: {sum(hit.distance for hit in results[0])/len(results[0]):.3f})"
        )
        
        # Format results to match orchestrator expectations
        formatted_results = []
        for hit in results[0]:
            try:
                entity_dict = hit.entity.to_dict()
                metadata = entity_dict.get("metadata", {})
                
                # Extract target_node from metadata
                target_node_data = metadata.get("target_node", {})
                
                formatted_result = {
                    "entity": {
                        "mapping_name": entity_dict.get("template_name", ""),
                        "resource_id": entity_dict.get("resourceId", ""),
                        "target_node_path": entity_dict.get("text", ""),
                        "target_node": {
                            "name": target_node_data.get("name", ""),
                            "value": target_node_data.get("value", ""),
                            "type": target_node_data.get("type", ""),
                            "code_value": target_node_data.get("code_value"),
                            "node_condition": target_node_data.get("node_condition"),
                            "loop_iterator": target_node_data.get("loop_iterator"),
                            "loop_conditions": target_node_data.get("loop_conditions", []),
                            "references": target_node_data.get("references", []),
                        },
                        "chunk_type": entity_dict.get("chunk_type", ""),
                        "pattern_type": entity_dict.get("pattern_type", ""),
                        "complexity_score": entity_dict.get("complexity_score", 0.0),
                    },
                    "distance": float(hit.distance),
                }
                
                formatted_results.append(formatted_result)
                
            except Exception as e:
                logger.warning(f"Error formatting search result: {e}")
                continue
        
        return formatted_results
        
    except Exception as e:
        logger.error(f"Failed to search in '{collection_name}': {str(e)}", exc_info=True)
        return []



def create_mapping_collections(dimension=768):  # Changed default from 384
    """Create both enhanced mapping collections with 768-dim embeddings"""
    collections = {}
    collection_names = ["OTMToJT", "JTToFedex"]

    for collection_name in collection_names:
        try:
            collection = create_collection(collection_name, dimension)
            if collection is not None:
                collections[collection_name] = collection
                logger.info(
                    f"✅ Successfully created/accessed enhanced 768-dim '{collection_name}' collection"
                )
            else:
                logger.error(
                    f"❌ Failed to create 768-dim '{collection_name}' collection"
                )

        except Exception as e:
            logger.error(
                f"❌ Error creating 768-dim '{collection_name}' collection: {str(e)}"
            )

    return collections
