#!/usr/bin/env python3
"""
Regenerate training wrapper scripts with correct paths.
"""

import sys
from pathlib import Path

# Add the parent directory to path so we can import our modules
parent_dir = Path(__file__).parent
sys.path.insert(0, str(parent_dir))

from enhanced_hyperopt_runner import EnhancedHyperOptRunner


def regenerate_wrappers():
    """Regenerate all training wrapper scripts with correct paths."""

    runner = EnhancedHyperOptRunner()
    models = ["ginny", "sigma", "pyggy", "attentivefp"]

    print("Regenerating training wrapper scripts...")

    for model in models:
        try:
            wrapper_path = runner.create_training_wrapper_script(model)
            print(f"✅ Regenerated wrapper for {model}: {wrapper_path}")
        except Exception as e:
            print(f"❌ Failed to regenerate wrapper for {model}: {e}")

    print("Done!")


if __name__ == "__main__":
    regenerate_wrappers()
