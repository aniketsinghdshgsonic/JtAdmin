# app/services/prompt_engineering.py - Enhanced prompts for CodeLlama-7B
import json
import re
from typing import Dict, Any, List

class EnhancedPromptBuilder:
    """Enhanced prompt builder optimized for CodeLlama-7B-Instruct"""
    
    @staticmethod
    def build_structured_mapping_prompt(
        parsed_input: Dict[str, Any], 
        vector_context: Dict[str, Any],
        complexity_level: str = "medium"
    ) -> str:
        """Build structured prompt optimized for CodeLlama-7B"""
        
        # Extract key info
        awb = parsed_input.get("refNum", "")
        origin = parsed_input.get("origin", "")
        destination = parsed_input.get("destination", "")
        
        # Get shipment details
        shipments = parsed_input.get("shipments", [{}])
        events_count = len(shipments[0].get("events", [])) if shipments else 0
        routes_count = len(shipments[0].get("routes", [])) if shipments else 0
        
        # Build context section
        context_section = ""
        if vector_context.get("templates"):
            best_template = vector_context["templates"][0]
            similarity = best_template.get("similarity", 0)
            
            if similarity > 0.4:  # Only use if good similarity
                context_section = f"""
[CONTEXT FROM SIMILAR MAPPING - Similarity: {similarity:.2f}]
{best_template.get('content', '')[:400]}
[END CONTEXT]

Use this as reference for generating the new mapping.
"""

        if complexity_level == "simple":
            return EnhancedPromptBuilder._build_simple_prompt(awb, origin, destination, context_section)
        elif complexity_level == "medium":
            return EnhancedPromptBuilder._build_medium_prompt(parsed_input, context_section)
        else:
            return EnhancedPromptBuilder._build_complex_prompt(parsed_input, context_section)

    @staticmethod
    def _build_simple_prompt(awb: str, origin: str, destination: str, context: str) -> str:
        """Simple prompt for basic field mapping"""
        return f"""[INST] You are a JSON mapping expert. Create a field mapping for air cargo data.

TASK: Generate JSON mapping for AWB {awb} from {origin} to {destination}

{context}

REQUIRED OUTPUT FORMAT (EXACT JSON):
{{
  "name": "Generated_Mapping_{awb.replace('-', '_') if awb else 'Unknown'}",
  "targetTreeNode": {{
    "name": "root",
    "children": [
      {{
        "name": "masterAirWayBillNumber",
        "value": "var1",
        "references": [{{
          "jsonId": 1,
          "path": "root.refNum",
          "var": "var1", 
          "text": true
        }}]
      }},
      {{
        "name": "trackingNumber",
        "value": "var2",
        "codeValue": "var2 = var2.toString().replace('-', '')",
        "references": [{{
          "jsonId": 2,
          "path": "root.refNum",
          "var": "var2",
          "text": true
        }}]
      }},
      {{
        "name": "origin", 
        "value": "var3",
        "references": [{{
          "jsonId": 3,
          "path": "root.origin",
          "var": "var3",
          "text": true
        }}]
      }},
      {{
        "name": "destination",
        "value": "var4", 
        "references": [{{
          "jsonId": 4,
          "path": "root.destination",
          "var": "var4",
          "text": true
        }}]
      }},
      {{
        "name": "mode",
        "value": "Freight1",
        "plain": true
      }}
    ]
  }},
  "localContext": {{
    "classes": [],
    "functions": [],
    "globalVariables": [],
    "lookupTables": []
  }}
}}

Generate ONLY the JSON above. Ensure all brackets and quotes are properly closed. [/INST]"""

    @staticmethod
    def _build_medium_prompt(parsed_input: Dict[str, Any], context: str) -> str:
        """Medium complexity prompt with more fields"""
        
        awb = parsed_input.get("refNum", "")
        origin = parsed_input.get("origin", "")
        destination = parsed_input.get("destination", "")
        
        # Extract weight info
        weight_value = ""
        weight_unit = ""
        pieces = ""
        
        additional_info = parsed_input.get("additionalInfo", {})
        if additional_info:
            weight = additional_info.get("weight", {})
            weight_value = weight.get("value", "") if weight else ""
            weight_unit = weight.get("unit", "") if weight else ""
            pieces = str(additional_info.get("totalPieces", "")) if additional_info.get("totalPieces") else ""

        return f"""[INST] You are an expert at generating JSON field mappings for air cargo systems.

TASK: Create a comprehensive field mapping for AWB {awb}

INPUT SUMMARY:
- AWB: {awb}
- Route: {origin} → {destination}  
- Weight: {weight_value} {weight_unit}
- Pieces: {pieces}

{context}

GENERATE THIS EXACT JSON STRUCTURE:
{{
  "name": "Medium_Mapping_{awb.replace('-', '_') if awb else 'Unknown'}",
  "targetTreeNode": {{
    "name": "root",
    "children": [
      {{
        "name": "masterAirWayBillNumber",
        "value": "var1",
        "references": [{{
          "jsonId": 1,
          "path": "root.refNum", 
          "var": "var1",
          "text": true
        }}]
      }},
      {{
        "name": "trackingNumber",
        "value": "var2",
        "codeValue": "var2 = var2.toString().replace('-', '')",
        "references": [{{
          "jsonId": 2,
          "path": "root.refNum",
          "var": "var2", 
          "text": true
        }}]
      }},
      {{
        "name": "origin",
        "value": "var3",
        "references": [{{
          "jsonId": 3,
          "path": "root.origin",
          "var": "var3",
          "text": true
        }}]
      }},
      {{
        "name": "destination",
        "value": "var4",
        "references": [{{
          "jsonId": 4,
          "path": "root.destination", 
          "var": "var4",
          "text": true
        }}]
      }},
      {{
        "name": "mode",
        "value": "Freight1",
        "plain": true
      }},
      {{
        "name": "totalPieces",
        "value": "var5",
        "references": [{{
          "jsonId": 5,
          "path": "root.additionalInfo.totalPieces",
          "var": "var5",
          "text": true
        }}]
      }},
      {{
        "name": "totalWeight",
        "children": [
          {{
            "name": "value",
            "value": "var6",
            "references": [{{
              "jsonId": 6,
              "path": "root.additionalInfo.weight.value",
              "var": "var6",
              "text": true
            }}]
          }},
          {{
            "name": "unit", 
            "value": "var7",
            "references": [{{
              "jsonId": 7,
              "path": "root.additionalInfo.weight.unit",
              "var": "var7",
              "text": true
            }}]
          }}
        ]
      }}
    ]
  }},
  "localContext": {{
    "classes": [],
    "functions": [],
    "globalVariables": [],
    "lookupTables": []
  }}
}}

OUTPUT: Generate only the JSON above. Ensure proper syntax. [/INST]"""

    @staticmethod
    def _build_complex_prompt(parsed_input: Dict[str, Any], context: str) -> str:
        """Complex prompt for advanced mappings"""
        
        awb = parsed_input.get("refNum", "")
        origin = parsed_input.get("origin", "")
        destination = parsed_input.get("destination", "")
        
        shipments = parsed_input.get("shipments", [{}])
        events_count = len(shipments[0].get("events", [])) if shipments else 0
        routes_count = len(shipments[0].get("routes", [])) if shipments else 0

        return f"""[INST] You are an expert at generating complex JSON field mappings for air cargo tracking systems.

TASK: Generate comprehensive mapping for AWB {awb} with event and route processing

INPUT ANALYSIS:
- AWB: {awb}
- Route: {origin} → {destination}
- Events: {events_count} tracking events
- Routes: {routes_count} flight segments
- Requires: Array processing, event mapping, route handling

{context}

REQUIRED JSON STRUCTURE WITH ARRAYS:
{{
  "name": "Complex_Mapping_{awb.replace('-', '_') if awb else 'Unknown'}",
  "targetTreeNode": {{
    "name": "root",
    "children": [
      {{
        "name": "masterAirWayBillNumber",
        "value": "var1",
        "references": [{{
          "jsonId": 1,
          "path": "root.refNum",
          "var": "var1",
          "text": true
        }}]
      }},
      {{
        "name": "trackingNumber",
        "value": "var2", 
        "codeValue": "var2 = var2.toString().replace('-', '')",
        "references": [{{
          "jsonId": 2,
          "path": "root.refNum",
          "var": "var2",
          "text": true
        }}]
      }},
      {{
        "name": "origin",
        "value": "var3",
        "references": [{{
          "jsonId": 3,
          "path": "root.origin",
          "var": "var3",
          "text": true
        }}]
      }},
      {{
        "name": "destination",
        "value": "var4",
        "references": [{{
          "jsonId": 4,
          "path": "root.destination",
          "var": "var4", 
          "text": true
        }}]
      }},
      {{
        "name": "mode",
        "value": "Freight1",
        "plain": true
      }},
      {{
        "name": "milestones",
        "type": "ar",
        "children": [
          {{
            "name": "[0]",
            "type": "ac",
            "looper": {{
              "loopStatement": "eventList.each"
            }},
            "children": [
              {{
                "name": "eventCode",
                "value": "eventCode"
              }},
              {{
                "name": "description", 
                "value": "description"
              }},
              {{
                "name": "station",
                "value": "station"
              }},
              {{
                "name": "eventTime",
                "value": "eventTime"
              }},
              {{
                "name": "sequence",
                "value": "sequence"
              }}
            ]
          }}
        ]
      }},
      {{
        "name": "routes",
        "type": "ar",
        "children": [
          {{
            "name": "[0]",
            "type": "ac", 
            "looper": {{
              "loopStatement": "routeList.each"
            }},
            "children": [
              {{
                "name": "origin",
                "value": "origin"
              }},
              {{
                "name": "destination",
                "value": "destination"
              }},
              {{
                "name": "flightNum",
                "value": "flightNum"
              }}
            ]
          }}
        ]
      }}
    ]
  }},
  "localContext": {{
    "classes": [],
    "functions": [],
    "globalVariables": [],
    "lookupTables": []
  }}
}}

Generate ONLY the JSON mapping above. [/INST]"""

    @staticmethod
    def build_json_repair_prompt(broken_json: str, error_message: str) -> str:
        """Build prompt to repair broken JSON"""
        return f"""[INST] Fix this broken JSON. The error was: {error_message}

BROKEN JSON:
{broken_json[:500]}

Fix the JSON syntax errors. Return only valid JSON:
[/INST]"""

    @staticmethod
    def build_chunk_prompt(field_name: str, source_path: str, var_name: str) -> str:
        """Build prompt for individual field mapping"""
        return f"""[INST] Generate JSON field mapping:

Field: {field_name}
Source: {source_path}
Variable: {var_name}

Format:
{{
  "name": "{field_name}",
  "value": "{var_name}",
  "references": [{{
    "jsonId": 1,
    "path": "{source_path}",
    "var": "{var_name}",
    "text": true
  }}]
}}

Generate JSON: [/INST]"""

class PromptValidator:
    """Validate and optimize prompts for CodeLlama"""
    
    @staticmethod
    def validate_prompt_length(prompt: str, max_tokens: int = 2000) -> bool:
        """Check if prompt is within token limits"""
        estimated_tokens = len(prompt.split()) * 1.3  # Rough estimate
        return estimated_tokens <= max_tokens
    
    @staticmethod
    def optimize_prompt_for_codellama(prompt: str) -> str:
        """Optimize prompt specifically for CodeLlama-7B-Instruct"""
        
        # Ensure proper instruction format
        if not prompt.startswith("[INST]"):
            prompt = f"[INST] {prompt}"
        
        if not prompt.endswith("[/INST]"):
            prompt = f"{prompt} [/INST]"
        
        # Add JSON validation hints
        if "JSON" in prompt and "valid" not in prompt.lower():
            prompt = prompt.replace("Generate", "Generate valid")
        
        # Limit length if too long
        if len(prompt) > 3000:
            # Truncate context but keep structure
            lines = prompt.split('\n')
            important_lines = []
            context_lines = []
            
            in_context = False
            for line in lines:
                if "[CONTEXT" in line:
                    in_context = True
                elif "[END CONTEXT]" in line:
                    in_context = False
                    continue
                elif in_context:
                    context_lines.append(line)
                else:
                    important_lines.append(line)
            
            # Keep only essential context
            if context_lines:
                context_lines = context_lines[:5]  # Limit context
                important_lines.insert(-2, "[CONTEXT]")
                important_lines.extend(context_lines)
                important_lines.insert(-1, "[END CONTEXT]")
            
            prompt = '\n'.join(important_lines)
        
        return prompt
    
    @staticmethod
    def extract_json_from_response(response: str) -> str:
        """Extract JSON from CodeLlama response"""
        
        # Remove instruction markers
        response = re.sub(r'\[/?INST\]', '', response)
        
        # Remove common prefixes
        response = re.sub(r'^[^{]*', '', response)
        response = re.sub(r'[^}]*$', '', response)
        
        # Find JSON block
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.finditer(json_pattern, response, re.DOTALL)
        
        for match in matches:
            candidate = match.group()
            if '"name"' in candidate and '"targetTreeNode"' in candidate:
                return candidate
        
        # Fallback: return largest JSON-like block
        if '{' in response and '}' in response:
            start = response.find('{')
            end = response.rfind('}') + 1
            return response[start:end]
        
        return response.strip()