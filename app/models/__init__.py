# app/models/__init__.py
"""
Models package for LLaMA Flask Application
"""

# Import main classes to make them available at package level
try:
    from .llama_manager import LlamaManager
except ImportError:
    LlamaManager = None

__all__ = ['LlamaManager']