"""Test script for virtual node functionality."""

import sys
import torch
import numpy as np
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from graphamole.data import Molecule
from GraphXSection_torch.adduct_features import AdductFeatureExtractor


def test_virtual_node_creation():
    """Test that virtual nodes are correctly added to molecular graphs."""
    print("=" * 80)
    print("Testing virtual node creation...")
    print("=" * 80)
    
    # Simple molecule
    smiles = "CCO"  # Ethanol
    mol = Molecule(smiles)
    
    # Without virtual node
    data_no_vnode = mol.get_data(add_virtual_node=False)
    print(f"\nWithout virtual node:")
    print(f"  Number of nodes: {data_no_vnode.x_atoms.size(0)}")
    print(f"  Number of edges: {data_no_vnode.edge_index.size(1)}")
    print(f"  Has virtual node flag: {getattr(data_no_vnode, 'has_virtual_node', False)}")
    print(f"  Node types: {data_no_vnode.node_types}")
    print(f"  Edge types: {data_no_vnode.edge_types}")
    
    # With virtual node (default zeros)
    data_with_vnode = mol.get_data(add_virtual_node=True)
    print(f"\nWith virtual node (zeros):")
    print(f"  Number of atom nodes: {data_with_vnode.x_atoms.size(0)}")
    print(f"  Virtual node features shape: {data_with_vnode.x_vnode.shape if data_with_vnode.x_vnode is not None else None}")
    print(f"  Total nodes: {data_with_vnode.x_atoms.size(0) + (1 if data_with_vnode.x_vnode is not None else 0)}")
    print(f"  Number of edges: {data_with_vnode.edge_index.size(1)}")
    print(f"  Has virtual node flag: {data_with_vnode.has_virtual_node}")
    print(f"  Virtual node index: {data_with_vnode.virtual_node_idx}")
    print(f"  Number of real atoms: {data_with_vnode.num_real_atoms}")
    print(f"  Node types: {data_with_vnode.node_types}")
    print(f"  Edge types (unique): {torch.unique(data_with_vnode.edge_types).tolist()}")
    
    # Check edges to virtual node
    virtual_idx = data_with_vnode.virtual_node_idx
    edges_to_virtual = (data_with_vnode.edge_index[1] == virtual_idx).sum().item()
    edges_from_virtual = (data_with_vnode.edge_index[0] == virtual_idx).sum().item()
    print(f"  Edges to virtual node: {edges_to_virtual}")
    print(f"  Edges from virtual node: {edges_from_virtual}")
    
    # Check edge types
    edge_types = data_with_vnode.edge_types
    bond_edges = (edge_types == 0).sum().item()
    adduct_edges = (edge_types == 1).sum().item()
    print(f"  Bond edges (type 0): {bond_edges}")
    print(f"  Adduct edges (type 1): {adduct_edges}")
    
    expected_edges = data_with_vnode.num_real_atoms
    assert edges_to_virtual == expected_edges, f"Expected {expected_edges} edges to virtual node"
    assert edges_from_virtual == expected_edges, f"Expected {expected_edges} edges from virtual node"
    assert adduct_edges == expected_edges * 2, f"Expected {expected_edges * 2} adduct edges"
    
    print("\n✓ Virtual node creation test passed!")
    return True


def test_adduct_feature_extraction():
    """Test adduct feature extractor."""
    print("\n" + "=" * 80)
    print("Testing adduct feature extraction...")
    print("=" * 80)
    
    extractor = AdductFeatureExtractor(feature_dim=16)
    
    # Test different adducts
    test_adducts = [
        "(+H)",
        "(-H)",
        "(+Na)",
        "(+K)",
        "(+2H)",
        "(-2H)",
    ]
    
    for adduct in test_adducts:
        features = extractor.extract_features(adduct)
        print(f"\n{adduct}:")
        print(f"  Feature shape: {features.shape}")
        print(f"  Feature dtype: {features.dtype}")
        print(f"  Non-zero features: {np.count_nonzero(features)}")
        
        # Parse for details
        parsed = extractor.parse_adduct(adduct)
        print(f"  Parsed charge: {parsed['charge']}")
        print(f"  Parsed ions: {parsed['ions']}")
        print(f"  Mass delta: {parsed['mass_delta']:.3f} Da")
        
        assert features.shape == (16,), f"Expected shape (16,), got {features.shape}"
        assert features.dtype == np.int64, f"Expected dtype int64, got {features.dtype}"
    
    print("\n✓ Adduct feature extraction test passed!")
    return True


def test_virtual_node_with_features():
    """Test virtual node with custom features."""
    print("\n" + "=" * 80)
    print("Testing virtual node with custom adduct features...")
    print("=" * 80)
    
    smiles = "CCOC"  # Methoxyethane
    mol = Molecule(smiles)
    
    # Extract adduct features
    extractor = AdductFeatureExtractor(feature_dim=16)
    adduct_features = extractor.extract_features("(+Na)")
    
    print(f"\nAdduct: (+Na)")
    print(f"Feature vector shape: {adduct_features.shape}")
    print(f"Feature vector (first 8): {adduct_features[:8]}")
    
    # Create graph with virtual node
    data = mol.get_data(
        add_virtual_node=True,
        virtual_node_features=adduct_features,
    )
    
    print(f"\nGraph with adduct virtual node:")
    print(f"  Atom nodes: {data.x_atoms.size(0)}")
    print(f"  Virtual node features shape: {data.x_vnode.shape}")
    print(f"  Virtual node features (first 8): {data.x_vnode[0][:8] if data.x_vnode.size(0) > 0 else 'N/A'}")
    print(f"  Virtual node is connected: {(data.edge_index[0] == data.virtual_node_idx).any().item()}")
    print(f"  Node types: {data.node_types}")
    print(f"  Number of edge types: {torch.unique(data.edge_types).numel()}")
    
    # Verify virtual node has correct features (raw, no padding!)
    virtual_features = data.x_vnode[0].numpy()
    assert np.array_equal(virtual_features, adduct_features), \
        f"Virtual node features don't match: {virtual_features} vs {adduct_features}"
    
    print(f"  Virtual node features correctly set (raw, no padding)")
    
    print("\n✓ Virtual node with custom features test passed!")
    return True


def test_graphxsection_dataset():
    """Test GraphXSectionDataset with virtual nodes."""
    print("\n" + "=" * 80)
    print("Testing GraphXSectionDataset with virtual nodes...")
    print("=" * 80)
    
    from GraphXSection_torch.data_extract import GraphXSectionDataset, AdductEncoder
    import pandas as pd
    
    # Create simple dataset
    smiles_list = ["CCO", "CCOC", "CCOCC"]
    adduct_list = ["(+H)", "(+Na)", "(+K)"]
    
    # Create adduct encoder (needs pandas Series)
    encoder = AdductEncoder()
    encoder.fit([pd.Series(adduct_list)])
    adduct_ohe = encoder.transform(adduct_list)
    
    targets = np.array([100.0, 150.0, 200.0])
    
    # Create dataset WITHOUT virtual nodes
    print("\nDataset without virtual nodes:")
    dataset_no_vnode = GraphXSectionDataset(
        smiles_list,
        adduct_list,
        adduct_ohe,
        targets,
        use_adduct_virtual_node=False,
    )
    
    sample = dataset_no_vnode[0]
    print(f"  First graph atom nodes: {sample['graph'].x_atoms.size(0)}")
    print(f"  Has virtual node: {getattr(sample['graph'], 'has_virtual_node', False)}")
    print(f"  Node types: {sample['graph'].node_types}")
    
    # Create dataset WITH virtual nodes
    print("\nDataset with virtual nodes:")
    dataset_vnode = GraphXSectionDataset(
        smiles_list,
        adduct_list,
        adduct_ohe,
        targets,
        use_adduct_virtual_node=True,
        adduct_feature_dim=16,
    )
    
    sample = dataset_vnode[0]
    print(f"  First graph atom nodes: {sample['graph'].x_atoms.size(0)}")
    print(f"  First graph virtual node: {sample['graph'].x_vnode.shape if sample['graph'].x_vnode is not None else None}")
    print(f"  Has virtual node: {sample['graph'].has_virtual_node}")
    print(f"  Virtual node idx: {sample['graph'].virtual_node_idx}")
    print(f"  Node types: {sample['graph'].node_types}")
    print(f"  Edge types (unique): {torch.unique(sample['graph'].edge_types).tolist()}")
    
    # Test all samples
    for i, (smiles, adduct) in enumerate(zip(smiles_list, adduct_list)):
        sample = dataset_vnode[i]
        assert sample['graph'].has_virtual_node, f"Sample {i} missing virtual node"
        assert sample['graph'].x_vnode is not None, f"Sample {i} missing virtual node features"
        assert sample['graph'].x_vnode.size(1) == 16, f"Sample {i} virtual node has wrong feature dim"
        print(f"  Sample {i} ({adduct}): {sample['graph'].x_atoms.size(0)} atoms, vnode shape {sample['graph'].x_vnode.shape}")
    
    print("\n✓ GraphXSectionDataset test passed!")
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("VIRTUAL NODE IMPLEMENTATION TESTS")
    print("=" * 80)
    
    tests = [
        test_virtual_node_creation,
        test_adduct_feature_extraction,
        test_virtual_node_with_features,
        test_graphxsection_dataset,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"\n✗ Test failed: {test.__name__}")
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 80)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 80)
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
