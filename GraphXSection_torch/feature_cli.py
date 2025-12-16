#!/usr/bin/env python3

"""CLI utility for feature configuration management in GraphXSection_torch."""

import click
import logging
from pathlib import Path
from rich.logging import RichHandler
from rich.console import Console
import sys
import os

# Add GraphXSection_torch to path
sys.path.append(os.path.dirname(__file__))

try:
    from GraphXSection_torch.feature_config import (
        FeatureConfigManager,
        create_feature_config_template,
    )
except ImportError:
    print("Could not import feature configuration modules. Please check installation.")
    sys.exit(1)

console = Console()
logger = logging.getLogger(__name__)


def setup_logging():
    """Setup logging for the CLI."""
    logging.basicConfig(
        format="%(message)s",
        level=logging.INFO,
        handlers=[RichHandler(console=console, show_path=False)],
    )


@click.group()
def cli():
    """GraphXSection_torch Feature Configuration Management CLI."""
    setup_logging()


@cli.command()
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    default="feature_config_template.json",
    help="Output path for the template file",
)
def create_template(output: str):
    """Create a template feature configuration file with default vocabularies."""
    output_path = Path(output)

    try:
        create_feature_config_template(output_path)
        console.print(f"✅ Template created at: {output_path}", style="green")
        console.print(
            "\\nEdit this file to customize vocabularies and features:", style="blue"
        )
        console.print(f"  - Modify 'vocabularies' to change token sets")
        console.print(
            f"  - Modify 'atom_features'/'bond_features' to change which features are used"
        )
        console.print(f"  - Set 'oov': false to disable out-of-vocabulary handling")

    except Exception as e:
        console.print(f"❌ Failed to create template: {e}", style="red")
        sys.exit(1)


@cli.command()
@click.argument("config_path", type=click.Path(exists=True))
def validate(config_path: str):
    """Validate a feature configuration file."""
    config_path = Path(config_path)

    try:
        manager = FeatureConfigManager(config_path)
        vocab_summary = manager.get_vocab_summary()

        console.print(f"✅ Configuration is valid: {config_path}", style="green")
        console.print("\\nVocabulary Summary:", style="blue")

        for feat_name, vocab_size in vocab_summary.items():
            console.print(f"  {feat_name}: {vocab_size} tokens")

        console.print(f"\\nTotal atom features: {len(manager.atom_featurizers)}")
        console.print(f"Total bond features: {len(manager.bond_featurizers)}")

    except Exception as e:
        console.print(f"❌ Configuration validation failed: {e}", style="red")
        sys.exit(1)


@cli.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.option(
    "-n",
    "--num-molecules",
    type=int,
    default=5,
    help="Number of molecules to test encoding",
)
def test(config_path: str, num_molecules: int):
    """Test feature configuration by encoding sample molecules."""
    config_path = Path(config_path)

    # Sample SMILES for testing
    test_smiles = [
        "CCO",  # Ethanol
        "CC(=O)O",  # Acetic acid
        "c1ccccc1",  # Benzene
        "CC(C)O",  # Isopropanol
        "CC(=O)NC1=CC=CC=C1",  # Acetanilide
        "CC1=CC=CC=C1",  # Toluene
        "CCN(CC)CC",  # Triethylamine
    ][:num_molecules]

    try:
        from graphamole.data import Molecule

        manager = FeatureConfigManager(config_path)
        featurizers = manager.get_featurizer_collection()

        console.print(f"🧪 Testing with {len(test_smiles)} molecules...", style="blue")

        success_count = 0
        for i, smiles in enumerate(test_smiles):
            try:
                mol = Molecule(smiles)
                graph = mol.get_data(featurizers)

                console.print(
                    f"  ✅ {smiles}: nodes={graph.x.shape[0]}, edges={graph.edge_index.shape[1]}"
                )
                console.print(
                    f"     Node features: {graph.x.shape}, Edge features: {graph.edge_attr.shape}"
                )
                success_count += 1

            except Exception as e:
                console.print(f"  ❌ {smiles}: {e}", style="red")

        console.print(
            f"\\n📊 Results: {success_count}/{len(test_smiles)} molecules encoded successfully"
        )

        if success_count == len(test_smiles):
            console.print(
                "🎉 All tests passed! Configuration is ready for use.", style="green"
            )
        else:
            console.print(
                "⚠️  Some molecules failed. Check vocabulary coverage.", style="yellow"
            )

    except Exception as e:
        console.print(f"❌ Test failed: {e}", style="red")
        sys.exit(1)


@cli.command()
def examples():
    """Show example feature configuration snippets."""
    console.print("📚 Feature Configuration Examples", style="bold blue")

    console.print(
        "\\n1. Minimal Configuration (only basic atom/bond types):", style="green"
    )
    console.print(
        """
{
  "vocabularies": {
    "AtomType": ["C", "N", "O", "S", "P"],
    "BondType": ["single", "double", "triple", "aromatic"]
  },
  "atom_features": [
    {"type": "AtomType", "oov": true}
  ],
  "bond_features": [
    {"type": "BondType", "oov": true}
  ]
}"""
    )

    console.print("\\n2. Extended Configuration (more features):", style="green")
    console.print(
        """
{
  "vocabularies": {
    "AtomType": ["C", "N", "O", "S", "P", "F", "Cl", "Br", "I"],
    "Hybridization": ["s", "sp", "sp2", "sp3"],
    "FormalCharge": [-2, -1, 0, 1, 2]
  },
  "atom_features": [
    {"type": "AtomType", "oov": true},
    {"type": "Hybridization", "oov": false},
    {"type": "FormalCharge", "oov": true}
  ],
  "bond_features": [
    {"type": "BondType", "oov": true},
    {"type": "Stereo", "oov": true}
  ]
}"""
    )

    console.print("\\n3. Organic Chemistry Focus:", style="green")
    console.print(
        """
{
  "vocabularies": {
    "AtomType": ["H", "C", "N", "O", "S", "P", "F", "Cl", "Br", "I"],
    "Hybridization": ["sp", "sp2", "sp3"],
    "TotalNumHs": [0, 1, 2, 3],
    "Degree": [1, 2, 3, 4]
  },
  "atom_features": [
    {"type": "AtomType", "oov": false},
    {"type": "Hybridization", "oov": false},
    {"type": "TotalNumHs", "oov": true},
    {"type": "Degree", "oov": true},
    {"type": "Aromatic", "oov": false}
  ]
}"""
    )

    console.print("\\n💡 Tips:", style="yellow")
    console.print("  - Set 'oov': false for features with small, known vocabularies")
    console.print("  - Set 'oov': true for features that might have unexpected values")
    console.print("  - Smaller vocabularies = faster training, but less flexibility")
    console.print("  - Use 'validate' command to check your configuration")


if __name__ == "__main__":
    cli()
