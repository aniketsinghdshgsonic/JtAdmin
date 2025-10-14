# app/__init__.py
"""
Flask Application Package
Main application package for the LLaMA-powered user information system
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__description__ = "LLaMA-powered user information system with Flask"

# app/models/__init__.py
"""
Models Package
Contains all model-related classes and managers
"""

from app.models.llama_manager import LlamaManager  # ✅ Correct path

__all__ = ['LlamaManager']




from app.milvus_connection import connect_to_milvus  # ✅ Correct path

__all__ = ['connect_to_milvus']

# app/core/__init__.py
"""
Core Package
Contains core configuration and utility classes
"""

# This can be empty for now or contain utility functions