# app/services/embedding_service.py
import logging
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
import json
import numpy as np

logger = logging.getLogger(__name__)

class EmbeddingService:
    """Service for generating embeddings from JSON text"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize with sentence transformer model"""
        self.model_name = model_name
        self.model = None
        self.dimension = 768  # Default for all-MiniLM-L6-v2
        self._load_model()
    
    def _load_model(self):
        """Load the sentence transformer model"""
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            self.dimension = self.model.get_sentence_embedding_dimension()
            logger.info(f"Model loaded successfully, dimension: {self.dimension}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    
    def json_to_searchable_text(self, json_obj: Dict[str, Any]) -> str:
        """Convert JSON object to searchable text representation"""
        try:
            # Strategy: Create a text representation that captures:
            # 1. Field names and structure
            # 2. Field types and values  
            # 3. Nested relationships
            # 4. Array patterns
            
            def extract_features(obj, path="", depth=0):
                features = []
                
                if isinstance(obj, dict):
                    for key, value in obj.items():
                        current_path = f"{path}.{key}" if path else key
                        
                        # Add field path
                        features.append(f"field:{current_path}")
                        
                        # Add field type and value info
                        if isinstance(value, str):
                            features.append(f"string_field:{current_path}")
                            if len(value) < 50:  # Short strings might be important
                                features.append(f"value:{value}")
                        elif isinstance(value, (int, float)):
                            features.append(f"numeric_field:{current_path}")
                        elif isinstance(value, bool):
                            features.append(f"boolean_field:{current_path}")
                        elif isinstance(value, list):
                            features.append(f"array_field:{current_path}")
                            features.append(f"array_length:{len(value)}")
                        elif isinstance(value, dict):
                            features.append(f"object_field:{current_path}")
                        
                        # Recurse into nested structures
                        if depth < 5:  # Limit recursion depth
                            features.extend(extract_features(value, current_path, depth + 1))
                
                elif isinstance(obj, list):
                    features.append(f"array_at:{path}")
                    features.append(f"array_size:{len(obj)}")
                    
                    # Sample first few items for pattern recognition
                    for i, item in enumerate(obj[:3]):
                        item_path = f"{path}[{i}]"
                        features.extend(extract_features(item, item_path, depth + 1))
                
                return features
            
            # Extract features
            features = extract_features(json_obj)
            
            # Also include the raw JSON structure (truncated)
            json_str = json.dumps(json_obj, separators=(',', ':'))
            if len(json_str) > 1000:
                json_str = json_str[:1000] + "..."
            
            # Combine features with JSON structure
            text_representation = " ".join(features) + " " + json_str
            
            return text_representation
            
        except Exception as e:
            logger.error(f"Error converting JSON to searchable text: {e}")
            # Fallback to simple JSON string
            return json.dumps(json_obj, separators=(',', ':'))[:2000]
    
    def embed_json(self, json_obj: Dict[str, Any]) -> List[float]:
        """Generate embedding for a JSON object"""
        try:
            # Convert JSON to searchable text
            text = self.json_to_searchable_text(json_obj)
            
            # Generate embedding
            embedding = self.model.encode(text, convert_to_tensor=False)
            
            # Ensure it's a list of floats
            if isinstance(embedding, np.ndarray):
                embedding = embedding.tolist()
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            # Return zero vector as fallback
            return [0.0] * self.dimension
    
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for plain text"""
        try:
            embedding = self.model.encode(text, convert_to_tensor=False)
            
            if isinstance(embedding, np.ndarray):
                embedding = embedding.tolist()
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating text embedding: {e}")
            return [0.0] * self.dimension
    
    def similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings"""
        try:
            # Convert to numpy arrays
            a = np.array(embedding1)
            b = np.array(embedding2)
            
            # Calculate cosine similarity
            similarity = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
            
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0
    
    def get_dimension(self) -> int:
        """Get the embedding dimension"""
        return self.dimension
    
    def is_ready(self) -> bool:
        """Check if the embedding service is ready"""
        return self.model is not None