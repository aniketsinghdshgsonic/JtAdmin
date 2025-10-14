# app/services/json_mapping_service.py
import json
import logging
import re
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

class JSONMappingService:
    """Core service for JSON mapping automation using LLM + retrieval"""
    
    def __init__(self, llama_manager, milvus_service, embedding_service):
        self.llama_manager = llama_manager
        self.milvus_service = milvus_service
        self.embedding_service = embedding_service
        
    def store_mapping_pair(self, 
                          unmapped_json: Dict[str, Any],
                          mapped_json: Dict[str, Any],
                          domain: str = "general",
                          complexity: str = "medium",
                          description: str = "") -> Dict[str, Any]:
        """Store a mapping pair for future reference"""
        try:
            # Generate embedding for the unmapped JSON
            embedding = self.embedding_service.embed_json(unmapped_json)
            
            # Store in Milvus
            success = self.milvus_service.insert_mapping_pair(
                unmapped_embedding=embedding,
                unmapped_json=unmapped_json,
                mapped_json=mapped_json,
                domain=domain,
                complexity=complexity,
                description=description
            )
            
            return {
                "success": success,
                "message": "Mapping pair stored successfully" if success else "Failed to store mapping pair",
                "domain": domain,
                "complexity": complexity
            }
            
        except Exception as e:
            logger.error(f"Error storing mapping pair: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def search_similar_examples(self, 
                               unmapped_json: Dict[str, Any],
                               top_k: int = 5,
                               domain_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search for similar unmapped examples"""
        try:
            # Generate embedding for query
            query_embedding = self.embedding_service.embed_json(unmapped_json)
            
            # Search in Milvus
            results = self.milvus_service.search_similar_mappings(
                query_embedding=query_embedding,
                top_k=top_k,
                domain_filter=domain_filter
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching similar examples: {e}")
            return []
    
    def _build_mapping_prompt(self, 
                             unmapped_json: Dict[str, Any],
                             examples: List[Dict[str, Any]]) -> str:
        """Build the LLM prompt for mapping generation"""
        
        prompt = """You are a JSON mapping expert. Your task is to transform unmapped JSON input into properly mapped output.

MAPPING RULES:
1. "plain" fields: Copy the value directly to the target field
2. "constant" fields: Use the constant value literally  
3. "reference" fields: Preserve the reference as given
4. "codeValue" fields: Apply the transformation logic specified
5. Expand nested arrays and objects correctly
6. Never invent fields that don't exist in the input
7. Always return valid JSON as your final answer

EXAMPLES OF CORRECT MAPPINGS:
"""
        
        # Add examples
        for i, example in enumerate(examples, 1):
            prompt += f"\nEXAMPLE {i}:\n"
            prompt += f"UNMAPPED INPUT:\n{json.dumps(example['unmapped'], indent=2)}\n"
            prompt += f"MAPPED OUTPUT:\n{json.dumps(example['mapped'], indent=2)}\n"
            prompt += f"SIMILARITY: {example['similarity']:.3f}\n"
        
        prompt += f"""
NOW GENERATE THE MAPPING:
UNMAPPED INPUT:
{json.dumps(unmapped_json, indent=2)}

MAPPED OUTPUT (return ONLY valid JSON, no explanation):"""
        
        return prompt
    
    def _extract_json_from_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Extract valid JSON from LLM response"""
        try:
            # First try to parse the response directly
            response = response.strip()
            
            # Look for JSON-like content
            json_patterns = [
                r'\{.*\}',  # Standard JSON object
                r'\[.*\]',  # JSON array
            ]
            
            for pattern in json_patterns:
                matches = re.findall(pattern, response, re.DOTALL)
                for match in matches:
                    try:
                        parsed = json.loads(match)
                        return parsed
                    except json.JSONDecodeError:
                        continue
            
            # If no JSON found, try to clean up the response
            lines = response.split('\n')
            for i, line in enumerate(lines):
                if line.strip().startswith('{'):
                    # Try to parse from this line to the end
                    remaining = '\n'.join(lines[i:])
                    try:
                        parsed = json.loads(remaining)
                        return parsed
                    except json.JSONDecodeError:
                        continue
            
            # Last resort: try to parse each line
            for line in lines:
                line = line.strip()
                if line.startswith('{') and line.endswith('}'):
                    try:
                        parsed = json.loads(line)
                        return parsed
                    except json.JSONDecodeError:
                        continue
            
            return None
            
        except Exception as e:
            logger.error(f"Error extracting JSON from response: {e}")
            return None
    
    def _validate_mapping_quality(self, 
                                 unmapped_json: Dict[str, Any],
                                 mapped_json: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the quality of a generated mapping"""
        try:
            validation = {
                "valid": True,
                "issues": [],
                "confidence": 1.0,
                "field_coverage": 0.0
            }
            
            # Check if mapped_json is valid
            if not isinstance(mapped_json, dict):
                validation["valid"] = False
                validation["issues"].append("Mapped output is not a valid JSON object")
                validation["confidence"] = 0.0
                return validation
            
            # Calculate field coverage (simplified)
            def count_fields(obj):
                count = 0
                if isinstance(obj, dict):
                    count += len(obj)
                    for value in obj.values():
                        count += count_fields(value)
                elif isinstance(obj, list):
                    for item in obj:
                        count += count_fields(item)
                return count
            
            unmapped_fields = count_fields(unmapped_json)
            mapped_fields = count_fields(mapped_json)
            
            if unmapped_fields > 0:
                validation["field_coverage"] = min(1.0, mapped_fields / unmapped_fields)
            
            # Basic structural validation
            if len(mapped_json) == 0:
                validation["issues"].append("Mapped output is empty")
                validation["confidence"] *= 0.5
            
            # Confidence based on field coverage
            if validation["field_coverage"] < 0.5:
                validation["confidence"] *= 0.7
                validation["issues"].append("Low field coverage")
            
            return validation
            
        except Exception as e:
            logger.error(f"Error validating mapping quality: {e}")
            return {
                "valid": False,
                "issues": [f"Validation error: {str(e)}"],
                "confidence": 0.0,
                "field_coverage": 0.0
            }
    
    def generate_mapping(self, 
                        unmapped_json: Dict[str, Any],
                        domain: str = "general",
                        top_k: int = 3) -> Dict[str, Any]:
        """Generate mapped JSON from unmapped input using LLM + retrieval"""
        try:
            # Step 1: Retrieve similar examples
            similar_examples = self.search_similar_examples(
                unmapped_json=unmapped_json,
                top_k=top_k,
                domain_filter=domain if domain != "general" else None
            )
            
            if not similar_examples:
                logger.warning("No similar examples found, generating with minimal context")
            
            # Step 2: Build prompt
            prompt = self._build_mapping_prompt(unmapped_json, similar_examples)
            
            # Step 3: Generate response with LLM
            response = self.llama_manager.generate_response(
                prompt=prompt,
                max_tokens=2048,
                temperature=0.1,  # Low temperature for consistent output
                timeout=180
            )
            
            if not response:
                raise Exception("LLM did not generate a response")
            
            # Step 4: Extract JSON from response
            mapped_json = self._extract_json_from_response(response)
            
            if mapped_json is None:
                raise Exception("Could not extract valid JSON from LLM response")
            
            # Step 5: Validate mapping quality
            validation = self._validate_mapping_quality(unmapped_json, mapped_json)
            
            # Step 6: Return result
            return {
                "success": True,
                "mapped_json": mapped_json,
                "confidence": validation["confidence"],
                "validation": validation,
                "examples_used": len(similar_examples),
                "best_similarity": similar_examples[0]["similarity"] if similar_examples else 0.0,
                "tokens_generated": len(response.split()) if response else 0,
                "raw_response": response[:500] + "..." if len(response) > 500 else response
            }
            
        except Exception as e:
            logger.error(f"Error generating mapping: {e}")
            return {
                "success": False,
                "error": str(e),
                "mapped_json": None,
                "confidence": 0.0
            }
    
    def validate_mapping(self, 
                        unmapped_json: Dict[str, Any],
                        mapped_json: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a mapping pair"""
        try:
            validation = self._validate_mapping_quality(unmapped_json, mapped_json)
            
            return {
                "valid": validation["valid"],
                "confidence": validation["confidence"],
                "field_coverage": validation["field_coverage"],
                "issues": validation["issues"],
                "recommendation": "Store as training example" if validation["valid"] and validation["confidence"] > 0.7 else "Review before storing"
            }
            
        except Exception as e:
            logger.error(f"Error validating mapping: {e}")
            return {
                "valid": False,
                "error": str(e),
                "confidence": 0.0
            }
    
    def generate_mapping_with_feedback(self, 
                                     unmapped_json: Dict[str, Any],
                                     domain: str = "general",
                                     max_attempts: int = 3) -> Dict[str, Any]:
        """Generate mapping with iterative improvement"""
        best_result = None
        best_confidence = 0.0
        
        for attempt in range(max_attempts):
            try:
                logger.info(f"Mapping attempt {attempt + 1}/{max_attempts}")
                
                # Adjust parameters for each attempt
                top_k = 3 + attempt  # Get more examples on retry
                temperature = 0.1 + (attempt * 0.05)  # Slightly increase creativity
                
                result = self.generate_mapping(
                    unmapped_json=unmapped_json,
                    domain=domain,
                    top_k=top_k
                )
                
                if result["success"]:
                    confidence = result["confidence"]
                    
                    if confidence > best_confidence:
                        best_result = result
                        best_confidence = confidence
                    
                    # If we get a high confidence result, use it
                    if confidence > 0.8:
                        logger.info(f"High confidence result achieved: {confidence:.3f}")
                        break
                
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed: {e}")
                continue
        
        if best_result:
            best_result["attempts_made"] = attempt + 1
            return best_result
        else:
            return {
                "success": False,
                "error": "All mapping attempts failed",
                "attempts_made": max_attempts
            }