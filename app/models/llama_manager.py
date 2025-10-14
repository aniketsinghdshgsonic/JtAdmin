# app/models/llama_manager.py - UPDATED WITH TRANSFORMATION HINT SUPPORT
import os
import logging
import time
import gc
import json
import re
import threading
from typing import Optional, Dict, Any
from llama_cpp import Llama
import torch

logger = logging.getLogger(__name__)


class LlamaManager:
    """Optimized LLama Manager with Enhanced JSON Generation and Transformation Hint Support"""

    def __init__(self):
        self.model = None
        self.model_path = None
        self.model_loaded = False

        # Background process tracking
        self._active_threads = set()
        self._shutdown_event = threading.Event()

        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        # OPTIMIZED SETTINGS for AWS g5.xlarge (24GB VRAM)
        self.model_params = {
            "model_path": os.getenv(
                "MODEL_PATH",
                os.getenv(
                    "LLAMA_MODEL_PATH", "/app/ai-model/codellama-13b.Q4_K_M.gguf"
                ),
            ),
            "n_ctx": 8192,
            "n_gpu_layers": 30,
            "n_batch": 512,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 40,
            "repeat_penalty": 1.1,
            "max_tokens": 2000,
            "verbose": False,
            "use_mmap": True,
            "use_mlock": False,
            "rope_scaling_type": 0,
            "rope_freq_base": 10000.0,
            "rope_freq_scale": 1.0,
            "f16_kv": True,
            "logits_all": False,
        }

        self.json_generation_params = {
            "max_tokens": 2000,
            "temperature": 0.4,  # INCREASED from 0.3 - prevents template echoing
            "top_p": 0.90,  # INCREASED from 0.85 - more diversity
            "top_k": 40,  # INCREASED from 30 - more options
            "repeat_penalty": 1.8,  # REDUCED from 2.0 - less aggressive
            "frequency_penalty": 0.3,  # REDUCED from 0.5
            "stop": ["</s>", "[/INST]", "\n]", "```"],
            "stream": False,
            "echo": False,
        }

        self._load_model()

        if self.model_loaded:
            self._verify_gpu_utilization()

    def _verify_gpu_utilization(self):
        """Verify model is properly loaded on GPU"""
        try:
            import torch

            if torch.cuda.is_available():
                import subprocess

                result = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=memory.used",
                        "--format=csv,nounits,noheader",
                    ],
                    encoding="utf-8",
                )
                gpu_mem_mb = float(result.strip())
                logger.info(f"🎮 GPU Memory Usage: {gpu_mem_mb:.0f} MB")

                if gpu_mem_mb < 3000:
                    logger.warning(
                        "⚠️ Low GPU memory usage - model may not be fully on GPU"
                    )
                else:
                    logger.info(
                        f"✅ Model successfully loaded on GPU ({self.model_params['n_gpu_layers']} layers)"
                    )
            else:
                logger.warning("⚠️ CUDA not available - model running on CPU")
        except Exception as e:
            logger.debug(f"Could not verify GPU usage: {e}")

    def force_cancel_all_operations(self):
        """Force cancel all ongoing LLM operations"""
        try:
            logger.info("🛑 FORCE CANCELLING all LLM operations...")
            self._shutdown_event.set()

            killed_count = 0
            for thread in list(self._active_threads):
                try:
                    if thread.is_alive():
                        thread._stop()
                        killed_count += 1
                except:
                    pass

            self._active_threads.clear()

            if hasattr(self, "_clear_cuda_cache"):
                self._clear_cuda_cache()

            logger.info(f"🛑 Killed {killed_count} background LLM threads")
            self._shutdown_event.clear()

        except Exception as e:
            logger.error(f"Error during force cancel: {e}")

    def _register_thread(self, thread):
        """Register a thread for tracking"""
        self._active_threads.add(thread)

    def _unregister_thread(self, thread):
        """Unregister a thread"""
        self._active_threads.discard(thread)

    def _load_model(self):
        """Load CodeLlama-13B model"""
        try:
            logger.info("Loading CodeLlama-13B model with optimized parameters...")
            start_time = time.time()

            if not os.path.exists(self.model_params["model_path"]):
                logger.error(f"Model file not found: {self.model_params['model_path']}")
                return False

            self.model = Llama(**self.model_params)
            self.model_path = self.model_params["model_path"]
            self.model_loaded = True

            load_time = time.time() - start_time
            logger.info(f"✅ CodeLlama-13B loaded in {load_time:.2f}s")

            self._test_performance()
            return True

        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}", exc_info=True)
            self.model = None
            self.model_loaded = False
            return False

    def _test_performance(self):
        """Test model performance"""
        try:
            test_prompt = '<s>[INST] Generate JSON: {"status": "ready"} [/INST]'
            start_time = time.time()

            response = self.model(
                test_prompt,
                max_tokens=50,
                temperature=0.1,
                stop=["</s>"],
                stream=False,
            )

            test_time = time.time() - start_time
            output_text = (
                response["choices"][0]["text"]
                if response and response.get("choices")
                else ""
            )
            tokens_generated = len(output_text.split()) if output_text else 0

            if tokens_generated > 0:
                tokens_per_second = tokens_generated / test_time
                logger.info(f"🚀 Performance: {tokens_per_second:.1f} tokens/sec")

                if tokens_per_second >= 20:
                    logger.info("✅ Excellent performance")
                elif tokens_per_second >= 10:
                    logger.info("✅ Good performance")
                else:
                    logger.warning("⚠️ Performance below expected - check GPU layers")

        except Exception as e:
            logger.warning(f"Performance test failed: {e}")

    def generate_response(
        self,
        prompt: str,
        max_tokens: int = 300,
        temperature: float = 0.3,
        timeout: int = 600,
        **kwargs,
    ) -> Optional[str]:
        """
        Generate simple text response (NOT JSON)
        Optimized for text mappings like: fieldName → source.path
        """

        # Validate model is loaded
        if not self.model:
            logger.error("❌ Model not loaded")
            return None

        # Safe generation parameters
        generation_params = {
            "max_tokens": min(max_tokens, 300),  # Cap for safety
            "temperature": temperature,
            "top_p": 0.85,
            "top_k": 40,
            "repeat_penalty": 1.3,
            "stop": ["</s>", "\n\n", "###"],  # Stop on double newline
            "stream": False,
            "echo": False,
        }

        # Clear CUDA cache
        self._clear_cuda_cache()

        try:
            # GPU safety check
            if torch.cuda.is_available():
                try:
                    allocated = torch.cuda.memory_allocated() / 1024**3
                    props = torch.cuda.get_device_properties(0)
                    total = props.total_memory / 1024**3
                    available = total - allocated

                    if available < 2.0:
                        logger.warning(f"⚠️ Low GPU memory: {available:.1f}GB")
                        torch.cuda.empty_cache()
                        gc.collect()
                except:
                    pass

            logger.debug(
                f"Generating text: max_tokens={generation_params['max_tokens']}, temp={temperature:.2f}"
            )

            # Generate text
            try:
                response = self.model(prompt, **generation_params)
            except RuntimeError as runtime_err:
                error_msg = str(runtime_err).lower()

                if "out of memory" in error_msg:
                    logger.error("🔴 GPU OUT OF MEMORY")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        gc.collect()
                    return None

                logger.error(f"Runtime error: {runtime_err}")
                raise

            # Validate response
            if not response or not response.get("choices"):
                logger.error("❌ Empty response from model")
                return None

            output_text = response["choices"][0]["text"].strip()

            # Log output
            logger.info(f"✅ Generated text ({len(output_text)} chars)")
            logger.debug(f"First 100 chars: {output_text[:100]}")

            # Basic validation
            if len(output_text) < 5:
                logger.warning("⚠️ Very short output")
                return None

            # Check for garbage
            if self._is_text_garbage(output_text):
                logger.error("❌ Garbage output detected")
                return None

            return output_text

        except KeyboardInterrupt:
            logger.warning("⚠️ Generation interrupted")
            return None

        except Exception as e:
            logger.error(f"❌ Generation failed: {e}", exc_info=True)

            # Emergency cleanup
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()
            except:
                pass

            return None

        finally:
            # Always cleanup
            try:
                self._clear_cuda_cache()
            except:
                pass

    def _is_text_garbage(self, text: str) -> bool:
        """Quick check for garbage text output"""

        if not text or len(text.strip()) < 5:
            return True

        # Pattern 1: Excessive special characters
        special_char_count = sum(
            1 for c in text if not c.isalnum() and not c.isspace() and c not in "→-.,;:"
        )
        if special_char_count > len(text) * 0.3:
            logger.error(f"Too many special chars: {special_char_count}")
            return True

        # Pattern 2: Repeating symbols
        if re.search(r"(.)\1{10,}", text):
            logger.error("Repeating characters detected")
            return True

        # Pattern 3: Unicode garbage
        non_ascii_count = sum(1 for c in text if ord(c) > 127)
        if non_ascii_count > 20:
            logger.error(f"Too many non-ASCII chars: {non_ascii_count}")
            return True

        return False

    def generate_json_response(
        self, prompt, max_tokens=500, temperature=0.4, timeout=180, **kwargs
    ):
        """
        Generate JSON or simple text response with crash prevention

        FIXES:
        - Prevents GPU memory crashes
        - Handles timeout properly
        - Better garbage detection
        - Crash recovery
        """

        # CRITICAL: Validate model is loaded
        if not self.model:
            logger.error("❌ Model not loaded - cannot generate")
            return ""

        # Generation parameters with safe defaults
        generation_params = {
            "max_tokens": min(max_tokens, 500),  # Cap at 500 to prevent crashes
            "temperature": temperature,
            "top_p": 0.90,
            "top_k": 40,
            "repeat_penalty": 1.5,
            "frequency_penalty": 0.2,
            "stop": ["</s>", "[/INST]", "\n]", "```", "..."],
            "stream": False,
            "echo": False,
        }

        # Clear CUDA cache BEFORE generation
        self._clear_cuda_cache()

        try:
            # Additional GPU safety
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

                    # Check available memory
                    props = torch.cuda.get_device_properties(0)
                    allocated = torch.cuda.memory_allocated() / 1024**3
                    total = props.total_memory / 1024**3
                    available = total - allocated

                    if available < 2.0:  # Less than 2GB available
                        logger.warning(f"⚠️ Low GPU memory: {available:.1f}GB available")
                        # Force aggressive cleanup
                        torch.cuda.empty_cache()
                        gc.collect()
                        torch.cuda.synchronize()

                except Exception as gpu_check_err:
                    logger.debug(f"GPU check failed: {gpu_check_err}")

            # Log generation attempt
            logger.debug(
                f"Generating: max_tokens={generation_params['max_tokens']}, temp={temperature:.2f}"
            )

            # SAFE GENERATION with proper error handling
            try:
                response = self.model(prompt, **generation_params)
            except RuntimeError as runtime_err:
                error_msg = str(runtime_err).lower()

                if "out of memory" in error_msg or "cuda" in error_msg:
                    logger.error("🔴 GPU OUT OF MEMORY - Clearing cache and aborting")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        gc.collect()
                    return ""

                elif "double free" in error_msg or "corruption" in error_msg:
                    logger.error("🔴 MEMORY CORRUPTION DETECTED - Aborting generation")
                    return ""

                else:
                    logger.error(f"Runtime error during generation: {runtime_err}")
                    raise

            # Validate response
            if not response or not response.get("choices"):
                logger.error("❌ Empty response from model")
                return ""

            output_text = response["choices"][0]["text"].strip()

            # Log raw output
            logger.info(f"🔍 RAW OUTPUT ({len(output_text)} chars)")
            logger.debug(f"First 200 chars: {output_text[:200]}")

            # CRITICAL VALIDATION: Detect garbage output
            if self._is_garbage_output(output_text):
                logger.error("❌ REJECTED: Garbage output detected")
                return ""

            # Check if output is simple text format (transformation hints)
            if "→" in output_text and "[" in output_text and "]" in output_text:
                logger.info("✅ Detected simple text format with transformation hints")
                lines = [l.strip() for l in output_text.split("\n") if "→" in l]
                if len(lines) >= 2:
                    logger.info(f"✅ Valid simple format with {len(lines)} mappings")
                    return output_text
                else:
                    logger.warning("⚠️ Too few mappings in simple format")
                    return ""

            # Try to extract JSON
            json_content = self._extract_json_content(output_text)

            if json_content:
                try:
                    parsed = json.loads(json_content)

                    # Validate proper var format
                    if self._contains_literal_values(parsed.get("children", [])):
                        logger.error(
                            "❌ REJECTED: Contains literal data values instead of var references"
                        )
                        return ""

                    # Validate sufficient mappings (reduced from 3 to 1 for flexibility)
                    children = parsed.get("children", [])
                    if len(children) < 1:
                        logger.warning(f"⚠️ Only {len(children)} mappings generated")
                        return ""

                    logger.info(
                        f"✅ Valid JSON with {len(children)} proper var-based mappings"
                    )
                    return json_content

                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON: {e}")
                    return ""

            logger.warning("Could not extract valid JSON or simple format")
            return ""

        except KeyboardInterrupt:
            logger.warning("⚠️ Generation interrupted by user")
            return ""

        except Exception as e:
            logger.error(f"❌ Generation failed: {e}", exc_info=True)

            # Emergency cleanup on any error
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()
            except:
                pass

            return ""

        finally:
            # Always cleanup after generation
            try:
                self._clear_cuda_cache()
            except:
                pass

    def _is_garbage_output(self, text: str) -> bool:
        """Detect garbage/malformed output patterns - ENHANCED"""

        if not text or len(text.strip()) < 10:
            return True

        # Pattern 1: Excessive unicode garbage
        unicode_garbage_count = sum(
            1
            for c in text
            if ord(c) > 127 and c not in "àáâãäåæçèéêëìíîïðñòóôõö÷øùúûüýþÿ"
        )
        if unicode_garbage_count > 20:
            logger.error(f"Detected {unicode_garbage_count} garbage Unicode chars")
            return True

        # Pattern 2: Repeating symbols (more than 10 in a row)
        if re.search(r"[^\w\s]{10,}", text):
            logger.error("Detected repeating symbols")
            return True

        # Pattern 3: Repeating digits in var names
        if re.search(r"var\d*(\d)\1{4,}", text):
            logger.warning("Detected repeating digits in var names")
            return True

        # Pattern 4: Unbalanced braces (more than 10 difference)
        brace_diff = abs(text.count("{") - text.count("}"))
        if brace_diff > 10:
            logger.error(f"Unbalanced braces: difference of {brace_diff}")
            return True

        # Pattern 5: Duplicate <s> tokens (LLM echoing prompt)
        if text.count("<s>") > 1:
            logger.warning("Detected duplicate <s> tokens (prompt echo)")
            return True

        # Pattern 6: Repeating words/phrases
        words = text.split()
        if len(words) > 10:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.3:  # Less than 30% unique words
                logger.error(f"Low unique word ratio: {unique_ratio:.2%}")
                return True

        # Pattern 7: Only punctuation/symbols
        alphanumeric_count = sum(1 for c in text if c.isalnum())
        if alphanumeric_count < len(text) * 0.5:  # Less than 50% alphanumeric
            logger.error("Too few alphanumeric characters")
            return True

        return False

    def _clear_cuda_cache(self):
        """Enhanced CUDA cache clearing with error handling"""
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
                torch.cuda.synchronize()
                logger.debug("✅ CUDA cache cleared")
        except Exception as e:
            logger.debug(f"CUDA cache clear failed: {e}")

    def _extract_json_content(self, text: str) -> str:
        """Extract JSON - handles both full objects and arrays, validates variable format"""

        # Remove instruction tokens
        text = text.replace("[/INST]", "").replace("[INST]", "").replace("<s>", "")

        if self._is_garbage_output(text):
            logger.error("❌ REJECTED: Garbage output in JSON extraction")
            return ""

        # Try to extract array first (most common LLM output format)
        array_pattern = r"\[\s*\{.*?\}\s*(?:,\s*\{.*?\}\s*)*\]"
        array_match = re.search(array_pattern, text, re.DOTALL)

        if array_match:
            candidate = array_match.group(0)
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, list) and len(parsed) > 0:
                    # Validate no literal data
                    if self._contains_literal_values(parsed):
                        logger.error("❌ REJECTED: Array contains literal data")
                        return ""

                    # Wrap array in proper structure
                    wrapped = {"name": "root", "children": parsed}
                    logger.info(f"✅ Extracted valid array with {len(parsed)} mappings")
                    return json.dumps(wrapped)
            except json.JSONDecodeError as e:
                logger.debug(f"Array JSON parse failed: {e}")

        # Try full object pattern
        json_pattern = r'\{\s*"name"\s*:\s*"root"\s*,\s*"children"\s*:\s*\[(.*?)\]\s*\}'
        match = re.search(json_pattern, text, re.DOTALL)

        if match:
            candidate = match.group(0)
            try:
                parsed = json.loads(candidate)

                if self._contains_literal_values(parsed.get("children", [])):
                    logger.error("❌ REJECTED: Object contains literal data")
                    return ""

                logger.info("✅ Extracted complete JSON structure")
                return candidate
            except json.JSONDecodeError as e:
                logger.debug(f"Full object JSON parse failed: {e}")

        # Fallback: Find balanced braces/brackets
        max_depth = 0
        current_depth = 0
        best_json = ""
        start_idx = -1

        for i, char in enumerate(text):
            if char == "{" or char == "[":
                if current_depth == 0:
                    start_idx = i
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif char == "}" or char == "]":
                current_depth -= 1
                if current_depth == 0 and start_idx != -1:
                    candidate = text[start_idx : i + 1]
                    try:
                        parsed = json.loads(candidate)
                        if len(candidate) > len(best_json):
                            best_json = candidate
                    except json.JSONDecodeError:
                        pass

        if best_json:
            try:
                parsed = json.loads(best_json)

                # Handle array - wrap it
                if isinstance(parsed, list):
                    if self._contains_literal_values(parsed):
                        logger.error(
                            "❌ REJECTED: Fallback array contains literal data"
                        )
                        return ""
                    wrapped = {"name": "root", "children": parsed}
                    logger.info(
                        f"✅ Extracted array with {len(parsed)} mappings (fallback)"
                    )
                    return json.dumps(wrapped)

                # Handle object
                elif isinstance(parsed, dict):
                    children = parsed.get("children", [])
                    if self._contains_literal_values(children):
                        logger.error(
                            "❌ REJECTED: Fallback object contains literal data"
                        )
                        return ""
                    logger.info("✅ Extracted object structure (fallback)")
                    return best_json

            except json.JSONDecodeError:
                pass

        logger.warning("❌ Could not extract valid JSON from LLM output")
        return ""

    def _contains_literal_values(self, mappings: list) -> bool:
        """STRICT validation: Check if mappings contain literal data values"""

        if not isinstance(mappings, list):
            return False

        for mapping in mappings:
            if not isinstance(mapping, dict):
                continue

            value = mapping.get("value", "")

            if not value:
                continue

            # VALID: Variable reference format (var123, var1, etc.)
            if isinstance(value, str) and re.match(r"^var\d+$", value):
                continue

            # VALID: Empty string or null
            if value == "" or value is None:
                continue

            # ❌ INVALID: Anything else is literal data
            if isinstance(value, str):
                # Reject if it contains spaces (likely descriptive text)
                if " " in value:
                    return True

                # Reject if it's longer than 20 chars (likely data)
                if len(value) > 20:
                    return True

                # Reject if it looks like a date
                if re.search(r"\d{2,4}[-/]\d{2}[-/]\d{2,4}", value):
                    return True

                # Reject if it's NOT "var" + number and has substantial content
                if not value.startswith("var") and len(value) > 5:
                    return True

        return False

    def format_instruct_prompt(self, instruction: str, context: str = "") -> str:
        """
        Format prompt for CodeLlama Instruct model

        Args:
            instruction: The main instruction
            context: Optional context/examples

        Returns:
            Properly formatted prompt
        """
        if context:
            return f"<s>[INST] {context}\n\n{instruction} [/INST]"
        else:
            return f"<s>[INST] {instruction} [/INST]"

    # 2. Add validation method:
    def validate_generation_quality(
        self, output: str, expected_fields: int = 0
    ) -> bool:
        """
        Validate LLM output quality

        Args:
            output: Generated text
            expected_fields: Expected number of field mappings

        Returns:
            True if quality is acceptable
        """
        if not output or len(output.strip()) < 50:
            logger.warning("Output too short")
            return False

        # Check for garbage patterns
        if self._is_garbage_output(output):
            return False

        # Check for JSON structure
        try:
            parsed = json.loads(output)
            if expected_fields > 0:
                children = parsed.get("children", [])
                if len(children) < expected_fields * 0.7:  # At least 70% of expected
                    logger.warning(f"Too few fields: {len(children)}/{expected_fields}")
                    return False
            return True
        except:
            # Not JSON - check if it's valid text format
            return "→" in output and "[" in output

        return True

    def _is_garbage_output(self, text: str) -> bool:
        """Detect garbage/malformed output patterns"""

        # Pattern 1: Unicode garbage
        unicode_garbage_count = sum(
            1 for c in text if ord(c) > 127 and c not in "àáâãäåæçèéêëìíîïðñòóôõö"
        )
        if unicode_garbage_count > 20:
            logger.error(f"Detected {unicode_garbage_count} garbage Unicode chars")
            return True

        # Pattern 2: Repeating symbols
        if re.search(r"[^\w\s]{10,}", text):
            logger.error("Detected repeating symbols")
            return True

        # Pattern 3: Repeating digits
        if re.search(r"var\d*(\d)\1{4,}", text):
            logger.warning("Detected repeating digits in var names")
            return True

        # Pattern 4: Malformed JSON indicators
        if text.count("{") > text.count("}") + 10:
            logger.error("Unbalanced braces")
            return True

        return False

    def _extract_json_content(self, text: str) -> str:
        """Extract JSON - handles both full objects and arrays, validates variable format"""

        # Remove instruction tokens
        text = text.replace("[/INST]", "").replace("[INST]", "")

        if self._is_garbage_output(text):
            logger.error(
                "❌ REJECTED: Garbage output detected (repeating chars/invalid vars)"
            )
            return ""

        # Try to extract array first (most common LLM output format)
        array_pattern = r"\[\s*\{.*?\}\s*(?:,\s*\{.*?\}\s*)*\]"
        array_match = re.search(array_pattern, text, re.DOTALL)

        if array_match:
            candidate = array_match.group(0)
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, list) and len(parsed) > 0:
                    # CRITICAL VALIDATION: Check for literal data values
                    if self._contains_literal_values(parsed):
                        logger.error(
                            "❌ REJECTED: Mappings contain literal data instead of variable references"
                        )
                        logger.error(
                            f"   Example bad value: {self._get_example_bad_value(parsed)}"
                        )
                        return ""

                    # Wrap array in proper structure
                    wrapped = {"name": "root", "children": parsed}
                    logger.info(f"✅ Extracted valid array with {len(parsed)} mappings")
                    return json.dumps(wrapped)
            except json.JSONDecodeError as e:
                logger.debug(f"Array JSON parse failed: {e}")

        # Try full object pattern with root/children structure
        json_pattern = r'\{\s*"name"\s*:\s*"root"\s*,\s*"children"\s*:\s*\[(.*?)\]\s*\}'
        match = re.search(json_pattern, text, re.DOTALL)

        if match:
            candidate = match.group(0)
            try:
                parsed = json.loads(candidate)

                # CRITICAL VALIDATION: Check for literal data values
                if self._contains_literal_values(parsed.get("children", [])):
                    logger.error(
                        "❌ REJECTED: Mappings contain literal data instead of variable references"
                    )
                    logger.error(
                        f"   Example bad value: {self._get_example_bad_value(parsed.get('children', []))}"
                    )
                    return ""

                logger.info("✅ Extracted complete JSON structure")
                return candidate
            except json.JSONDecodeError as e:
                logger.debug(f"Full object JSON parse failed: {e}")

        # Fallback: Find balanced braces/brackets
        max_depth = 0
        current_depth = 0
        best_json = ""
        start_idx = -1

        for i, char in enumerate(text):
            if char == "{" or char == "[":
                if current_depth == 0:
                    start_idx = i
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif char == "}" or char == "]":
                current_depth -= 1
                if current_depth == 0 and start_idx != -1:
                    candidate = text[start_idx : i + 1]
                    try:
                        parsed = json.loads(candidate)
                        if len(candidate) > len(best_json):
                            best_json = candidate
                    except json.JSONDecodeError:
                        pass

        if best_json:
            try:
                parsed = json.loads(best_json)

                # Handle array - wrap it
                if isinstance(parsed, list):
                    if self._contains_literal_values(parsed):
                        logger.error(
                            "❌ REJECTED: Mappings contain literal data instead of variable references"
                        )
                        return ""
                    wrapped = {"name": "root", "children": parsed}
                    logger.info(
                        f"✅ Extracted array with {len(parsed)} mappings (fallback)"
                    )
                    return json.dumps(wrapped)

                # Handle object - validate it
                elif isinstance(parsed, dict):
                    children = parsed.get("children", [])
                    if self._contains_literal_values(children):
                        logger.error(
                            "❌ REJECTED: Mappings contain literal data instead of variable references"
                        )
                        return ""
                    logger.info("✅ Extracted object structure (fallback)")
                    return best_json

            except json.JSONDecodeError:
                pass

        logger.warning("❌ Could not extract valid JSON from LLM output")
        return ""

    def _contains_literal_values(self, mappings: list) -> bool:
        """STRICT validation: Check if mappings contain literal data values"""

        if not isinstance(mappings, list):
            return False

        for mapping in mappings:
            if not isinstance(mapping, dict):
                continue

            value = mapping.get("value", "")

            if not value:
                continue

            # VALID: Variable reference format (var123, var1, etc.)
            if isinstance(value, str) and re.match(r"^var\d+$", value):
                continue

            # VALID: Empty string or null
            if value == "" or value is None:
                continue

            # ❌ INVALID: Anything else is literal data
            if isinstance(value, str):
                # Reject if it contains spaces (likely descriptive text)
                if " " in value:
                    logger.error(
                        f"❌ REJECTED: Value contains spaces (literal text): {value[:50]}"
                    )
                    return True

                # Reject if it's longer than 20 chars (likely data)
                if len(value) > 20:
                    logger.error(
                        f"❌ REJECTED: Value too long (literal data): {value[:50]}"
                    )
                    return True

                # Reject if it looks like a date
                if re.search(r"\d{2,4}[-/]\d{2}[-/]\d{2,4}", value):
                    logger.error(f"❌ REJECTED: Date pattern (literal data): {value}")
                    return True

                # Reject if it's NOT "var" + number and has substantial content
                if not value.startswith("var") and len(value) > 5:
                    logger.error(f"❌ REJECTED: Non-var value with content: {value}")
                    return True

        return False

    def _get_example_bad_value(self, mappings: list) -> str:
        """Get an example of a bad value for logging"""
        if not isinstance(mappings, list):
            return "N/A"

        for mapping in mappings[:5]:
            if not isinstance(mapping, dict):
                continue

            value = mapping.get("value", "")

            # Return first non-variable value found
            if value and not re.match(r"^var\d+$", str(value)):
                return str(value)[:100]  # Truncate for logging

        return "N/A"

    def _has_real_mappings(self, parsed_json):
        """Check if JSON has actual field mappings (not empty template)"""

        if not isinstance(parsed_json, dict):
            return False

        children = parsed_json.get("children", [])

        if len(children) == 0:
            return False

        # Check first 3 children for real data
        valid_count = 0
        for child in children[:3]:
            if not isinstance(child, dict):
                continue

            # Must have non-empty value
            value = child.get("value", "")
            if not value or value == "":
                continue

            # Must have references with actual paths
            refs = child.get("references", [])
            if not refs or len(refs) == 0:
                continue

            for ref in refs:
                path = ref.get("path", "")
                # Check it's a real path, not placeholder
                if path and path.startswith("root.") and len(path) > 6:
                    valid_count += 1
                    break

        # At least 2 out of first 3 should be valid
        return valid_count >= 2

    def _log_gpu_status(self):
        """Log GPU status"""
        try:
            logger.debug(f"🎮 GPU: {self.model_params['n_gpu_layers']} layers")
        except:
            pass

    def is_model_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.model_loaded and self.model is not None

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        if not self.is_model_loaded():
            return {"loaded": False, "error": "Model not loaded"}

        return {
            "loaded": True,
            "model_path": self.model_path,
            "model_type": "CodeLlama-13B-Instruct",
            "quantization": "Q4_K_M",
            "context_size": self.model_params["n_ctx"],
            "gpu_layers": self.model_params["n_gpu_layers"],
            "temperature": self.json_generation_params["temperature"],
            "optimized_for": "JSON generation with transformation hints support",
            "expected_speed": "15-30 tokens/sec",
        }

    def _clear_cuda_cache(self):
        """Clear CUDA cache"""
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.debug("CUDA cache cleared")
        except:
            pass

    def unload_model(self):
        """Unload GGUF model and free GPU memory"""
        try:
            logger.info("🧹 Unloading GGUF model from GPU...")

            # Cancel any ongoing operations
            self.force_cancel_all_operations()

            # Delete the model instance
            if self.model:
                del self.model
                self.model = None
                self.model_loaded = False
                logger.info("✅ Model instance deleted")

            # Force Python garbage collection
            gc.collect()

            # Clear CUDA cache if available
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    torch.cuda.ipc_collect()
                    logger.info("✅ CUDA cache cleared")

                    # Verify memory was freed
                    import subprocess

                    result = subprocess.check_output(
                        [
                            "nvidia-smi",
                            "--query-gpu=memory.used",
                            "--format=csv,nounits,noheader",
                        ],
                        encoding="utf-8",
                    )
                    gpu_mem_mb = float(result.strip())
                    logger.info(f"📊 GPU memory after unload: {gpu_mem_mb:.0f} MB")
            except Exception as cuda_error:
                logger.warning(f"CUDA cleanup warning: {cuda_error}")

            logger.info("✅ GGUF model unloaded successfully")
            return True

        except Exception as e:
            logger.error(f"Error unloading GGUF model: {e}", exc_info=True)
            return False

    def diagnose_performance_issues(self):
        """Diagnose performance problems"""
        logger.info("🔍 Running diagnostics...")

        issues = []

        if self.model_params["n_gpu_layers"] < 33:
            issues.append(
                {
                    "issue": "Too few GPU layers",
                    "current": self.model_params["n_gpu_layers"],
                    "recommended": 35,
                    "impact": "HIGH",
                }
            )

        if self.json_generation_params["temperature"] < 0.15:
            issues.append(
                {
                    "issue": "Temperature too low (causes template echoing)",
                    "current": self.json_generation_params["temperature"],
                    "recommended": 0.2,
                    "impact": "HIGH - causes LLM to repeat prompts",
                }
            )

        try:
            import torch

            if not torch.cuda.is_available():
                issues.append(
                    {
                        "issue": "CUDA not available",
                        "current": "CPU only",
                        "recommended": "GPU with CUDA",
                        "impact": "CRITICAL",
                    }
                )
            else:
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                logger.info(f"GPU: {gpu_name}, {gpu_memory:.1f}GB")

                if gpu_memory < 12:
                    issues.append(
                        {
                            "issue": "Low GPU memory",
                            "current": f"{gpu_memory:.1f}GB",
                            "recommended": "16GB+",
                            "impact": "MEDIUM",
                        }
                    )
        except ImportError:
            issues.append({"issue": "PyTorch not installed", "impact": "CRITICAL"})

        if issues:
            logger.warning(f"⚠️ Found {len(issues)} issues:")
            for i, issue in enumerate(issues, 1):
                logger.warning(f"  {i}. {issue['issue']}")
                logger.warning(f"     Impact: {issue['impact']}")
        else:
            logger.info("✅ No issues detected")

        return issues
