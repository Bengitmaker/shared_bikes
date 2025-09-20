"""
Documentation package initialization.
This directory contains project documentation and utilities.
"""

from .doc_generator import DocGenerator

# 便捷访问
generator = DocGenerator()

__all__ = ['DocGenerator', 'generator']