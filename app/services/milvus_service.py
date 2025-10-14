# app/services/milvus_service.py
import os
import logging
from typing import Dict, List, Any, Optional, Tuple
from pymilvus import Collection, connections, utility, FieldSchema, CollectionSchema, DataType
import json
import datetime

logger = logging.getLogger(__name__)

class MilvusService:
    """Service for managing Milvus vector database operations"""
    
    def __init__(self):
        self.host = os.getenv("MILVUS_HOST", "localhost")
        self.port = int(os.getenv("MILVUS_PORT", "19530"))
        self.connected = False
        self.collections = {}
        
    def connect(self) -> bool:
        """Connect to Milvus database"""
        try:
            connections.connect(
                alias="default",
                host=self.host,
                port=self.port
            )
            self.connected = True
            logger.info(f"Connected to Milvus at {self.host}:{self.port}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {e}")
            self.connected = False
            return False
    
    def disconnect(self):
        """Disconnect from Milvus"""
        try:
            connections.disconnect("default")
            self.connected = False
            logger.info("Disconnected from Milvus")
        except Exception as e:
            logger.error(f"Error disconnecting from Milvus: {e}")
    
    def is_connected(self) -> bool:
        """Check if connected to Milvus"""
        return self.connected
    
    def create_mapping_collections(self):
        """Create collections for storing JSON mapping pairs"""
        if not self.connected:
            raise Exception("Not connected to Milvus")
        
        # Define collection schema for mapping pairs
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="unmapped_embedding", dtype=DataType.FLOAT_VECTOR, dim=768),
            FieldSchema(name="unmapped_text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="mapped_text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="domain", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="complexity", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="description", dtype=DataType.VARCHAR, max_length=1000),
            FieldSchema(name="created_at", dtype=DataType.VARCHAR, max_length=50)
        ]
        
        schema = CollectionSchema(
            fields=fields,
            description="JSON mapping pairs for LLM training"
        )
        
        collection_name = "json_mapping_pairs"
        
        try:
            # Drop existing collection if it exists (for development)
            if utility.has_collection(collection_name):
                utility.drop_collection(collection_name)
                logger.info(f"Dropped existing collection: {collection_name}")
            
            # Create new collection
            collection = Collection(
                name=collection_name,
                schema=schema,
                using='default'
            )
            
            # Create index for vector search
            index_params = {
                "metric_type": "COSINE",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 128}
            }
            
            collection.create_index(
                field_name="unmapped_embedding",
                index_params=index_params
            )
            
            collection.load()
            self.collections[collection_name] = collection
            
            logger.info(f"Created and loaded collection: {collection_name}")
            
        except Exception as e:
            logger.error(f"Error creating collection {collection_name}: {e}")
            raise
    
    def insert_mapping_pair(self, 
                           unmapped_embedding: List[float],
                           unmapped_json: Dict[str, Any],
                           mapped_json: Dict[str, Any],
                           domain: str = "general",
                           complexity: str = "medium",
                           description: str = "") -> bool:
        """Insert a mapping pair into the collection"""
        try:
            collection = self.collections.get("json_mapping_pairs")
            if not collection:
                raise Exception("Collection not found")
            
            # Prepare data
            data = {
                "unmapped_embedding": [unmapped_embedding],
                "unmapped_text": [json.dumps(unmapped_json, separators=(',', ':'))],
                "mapped_text": [json.dumps(mapped_json, separators=(',', ':'))],
                "domain": [domain],
                "complexity": [complexity],
                "description": [description],
                "created_at": [str(datetime.now().isoformat())]
            }
            
            # Insert data
            collection.insert(data)
            collection.flush()
            
            logger.info(f"Inserted mapping pair for domain: {domain}")
            return True
            
        except Exception as e:
            logger.error(f"Error inserting mapping pair: {e}")
            return False
    
    def search_similar_mappings(self, 
                               query_embedding: List[float],
                               top_k: int = 5,
                               domain_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search for similar unmapped examples"""
        try:
            collection = self.collections.get("json_mapping_pairs")
            if not collection:
                raise Exception("Collection not found")
            
            # Build search expression
            search_params = {
                "metric_type": "COSINE",
                "params": {"nprobe": 10}
            }
            
            expr = None
            if domain_filter:
                expr = f'domain == "{domain_filter}"'
            
            # Perform search
            results = collection.search(
                data=[query_embedding],
                anns_field="unmapped_embedding",
                param=search_params,
                limit=top_k,
                expr=expr,
                output_fields=["unmapped_text", "mapped_text", "domain", "complexity", "description"]
            )
            
            # Process results
            similar_examples = []
            for hit in results[0]:
                try:
                    unmapped_json = json.loads(hit.entity.get("unmapped_text"))
                    mapped_json = json.loads(hit.entity.get("mapped_text"))
                    
                    similar_examples.append({
                        "unmapped": unmapped_json,
                        "mapped": mapped_json,
                        "similarity": hit.score,
                        "domain": hit.entity.get("domain"),
                        "complexity": hit.entity.get("complexity"),
                        "description": hit.entity.get("description")
                    })
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse JSON from search result: {e}")
                    continue
            
            logger.info(f"Found {len(similar_examples)} similar examples")
            return similar_examples
            
        except Exception as e:
            logger.error(f"Error searching similar mappings: {e}")
            return []
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics for all collections"""
        stats = {}
        
        try:
            for name, collection in self.collections.items():
                collection.load()
                stats[name] = {
                    "num_entities": collection.num_entities,
                    "loaded": True
                }
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            stats["error"] = str(e)
        
        return stats
    
    def delete_collection(self, collection_name: str) -> bool:
        """Delete a collection"""
        try:
            if utility.has_collection(collection_name):
                utility.drop_collection(collection_name)
                if collection_name in self.collections:
                    del self.collections[collection_name]
                logger.info(f"Deleted collection: {collection_name}")
                return True
            else:
                logger.warning(f"Collection {collection_name} does not exist")
                return False
                
        except Exception as e:
            logger.error(f"Error deleting collection {collection_name}: {e}")
            return False