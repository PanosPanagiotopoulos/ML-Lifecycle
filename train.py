#!/usr/bin/env python
"""
Main training entry point for LLM fine-tuning.

Usage:
    python train.py configs/train.yaml
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.training.llm_trainer import main

if __name__ == "__main__":
    main()
