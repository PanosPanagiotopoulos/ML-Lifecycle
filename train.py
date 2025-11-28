#!/usr/bin/env python
"""
Main training entry point.

Usage:
    python train.py configs/train.yaml
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.training.model_trainer import main

if __name__ == "__main__":
    main()
