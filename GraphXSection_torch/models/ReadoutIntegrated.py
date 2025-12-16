"""
Integrated readout layers that concatenate global features directly.
This matches TensorFlow MolGraph's *ReadoutAndConcatGlobal pattern.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from torch_geometric.utils import scatter
from typing import Optional, Dict, Any, Tuple
from graphamole.graphamole.layers.AttentiveFPReadout import AttentiveFPReadout


class NoInputLSTMCell(nn.Module):
    """
    Custom LSTM Cell that takes no input, matching MolGraph's implementation.
    
    This LSTM cell only uses the previous hidden state and cell state,
    without any external input at each timestep.
    """
    
    def __init__(self, memory_state_dim: int, carry_state_dim: int):
        super().__init__()
        
        self.memory_state_dim = memory_state_dim
        self.carry_state_dim = carry_state_dim
        
        # Recurrent kernel for transforming carry state
        self.recurrent_kernel = nn.Linear(carry_state_dim, memory_state_dim * 4, bias=False)
        
        # Bias for the 4 gates/states
        self.bias = nn.Parameter(torch.zeros(memory_state_dim * 4))
        
        # Initialize weights similar to TensorFlow
        nn.init.xavier_uniform_(self.recurrent_kernel.weight)
        
    def forward(self, memory_state: torch.Tensor, carry_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            memory_state: [batch_size, memory_state_dim]
            carry_state: [batch_size, carry_state_dim]
            
        Returns:
            Tuple of (new_memory_state, new_carry_state)
        """
        # Perform linear transformation on carry_state
        z = self.recurrent_kernel(carry_state) + self.bias
        
        # Split into 4 gates/states
        update_gate, forget_gate, memory_state_candidate, output_gate = torch.chunk(z, 4, dim=1)
        
        # Apply activations
        update_gate = torch.sigmoid(update_gate)
        forget_gate = torch.sigmoid(forget_gate)
        output_gate = torch.sigmoid(output_gate)
        memory_state_candidate = torch.tanh(memory_state_candidate)
        
        # Update memory state
        new_memory_state = forget_gate * memory_state + update_gate * memory_state_candidate
        
        # Update carry state
        new_carry_state = output_gate * torch.tanh(new_memory_state)
        
        return new_memory_state, new_carry_state


class SetGatherReadout(nn.Module):
    """
    Set-to-set layer for graph readout, matching MolGraph's implementation.
    
    Based on Gilmer et al. (2017) and Vinyals et al. (2016).
    Uses LSTM-based attention mechanism with multiple steps.
    """
    
    def __init__(self, steps: int = 8):
        super().__init__()
        self.steps = steps
        self.lstm_cell = None  # Will be created dynamically based on input dimensions
        
    def _create_lstm_cell(self, node_dim: int):
        """Create LSTM cell based on node feature dimensions"""
        if self.lstm_cell is None:
            # Memory state has same dim as nodes, carry state is doubled (concatenated)
            self.lstm_cell = NoInputLSTMCell(
                memory_state_dim=node_dim,
                carry_state_dim=node_dim * 2  # Will be concatenated with attention readout
            )
        
    def _softmax_by_segment(self, values: torch.Tensor, segment_ids: torch.Tensor) -> torch.Tensor:
        """
        Apply softmax within each segment (graph) separately.
        
        Args:
            values: [num_nodes] attention scores
            segment_ids: [num_nodes] batch assignment
            
        Returns:
            [num_nodes, 1] normalized attention coefficients
        """
        # Subtract max for numerical stability within each segment
        segment_max = scatter(values, segment_ids, dim=0, reduce='max')[segment_ids]
        values_stable = values - segment_max
        
        # Compute exp
        exp_values = torch.exp(values_stable)
        
        # Sum within each segment
        segment_sum = scatter(exp_values, segment_ids, dim=0, reduce='sum')[segment_ids]
        
        # Normalize and add dimension
        attention_coef = exp_values / segment_sum
        return attention_coef.unsqueeze(1)
        
    def forward(self, node_features: torch.Tensor, batch_idx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            node_features: [num_nodes, node_dim]
            batch_idx: [num_nodes] batch assignment
            
        Returns:
            [batch_size, node_dim * 2] (doubled output dimension like MolGraph)
        """
        device = node_features.device
        node_dim = node_features.size(1)
        num_graphs = batch_idx.max().item() + 1
        
        # Create LSTM cell if needed
        self._create_lstm_cell(node_dim)
        if self.lstm_cell.recurrent_kernel.weight.device != device:
            self.lstm_cell = self.lstm_cell.to(device)
        
        # Initialize states
        memory_state = torch.zeros(num_graphs, node_dim, device=device)
        carry_state = torch.zeros(num_graphs, node_dim, device=device)
        
        # Perform set-to-set steps
        for step in range(self.steps):
            # Expand carry state to match node features
            carry_state_expanded = carry_state[batch_idx]  # [num_nodes, node_dim]
            
            # Compute attention scores (dot product)
            scores = torch.sum(node_features * carry_state_expanded, dim=1)  # [num_nodes]
            
            # Apply softmax within each graph
            attention_coef = self._softmax_by_segment(scores, batch_idx)  # [num_nodes, 1]
            
            # Apply attention and aggregate within each graph
            weighted_features = attention_coef * node_features  # [num_nodes, node_dim]
            attention_readout = scatter(weighted_features, batch_idx, dim=0, reduce='sum')  # [batch_size, node_dim]
            
            # Concatenate carry state with attention readout
            carry_state_evolved = torch.cat([carry_state, attention_readout], dim=1)  # [batch_size, node_dim * 2]
            
            # LSTM step (no external input)
            memory_state, carry_state = self.lstm_cell(memory_state, carry_state_evolved)
            
        # Return the evolved carry state (doubled dimension)
        return carry_state_evolved


class SetGatherReadoutAndConcatGlobal(nn.Module):
    """
    Set-Gather readout with global feature concatenation.
    Mimics TensorFlow's SetGatherReadoutAndConcatGlobal with full LSTM implementation.
    """

    def __init__(self, steps: int = 8, **kwargs):
        super().__init__()
        self.readout = SetGatherReadout(steps=steps)

    def forward(
        self,
        node_features: torch.Tensor,
        batch_idx: torch.Tensor,
        global_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            node_features: [num_nodes, node_dim]
            batch_idx: [num_nodes] batch assignment
            global_features: [batch_size, global_dim] or None

        Returns:
            [batch_size, (node_dim * 2) + global_dim] if global_features provided
            [batch_size, node_dim * 2] otherwise (doubled like MolGraph)
        """
        # Apply SetGather readout (returns doubled dimension)
        pooled = self.readout(node_features, batch_idx)

        # Concatenate with global features if provided
        if global_features is not None:
            return torch.cat([pooled, global_features], dim=-1)
        return pooled


class SegmentPoolingReadoutAndConcatGlobal(nn.Module):
    """
    Segment pooling readout with global feature concatenation.
    Mimics TensorFlow's SegmentPoolingReadoutAndConcatGlobal.
    """

    def __init__(self, mode: str = "mean", **kwargs):
        super().__init__()
        if mode == "mean" or mode == "average" or mode == "avg":
            self.pool_fn = global_mean_pool
        elif mode == "sum":
            self.pool_fn = global_add_pool
        elif mode == "max":
            self.pool_fn = global_max_pool
        else:
            raise ValueError(f"Invalid pooling mode: {mode}")

    def forward(
        self,
        node_features: torch.Tensor,
        batch_idx: torch.Tensor,
        global_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            node_features: [num_nodes, node_dim]
            batch_idx: [num_nodes] batch assignment
            global_features: [batch_size, global_dim] or None

        Returns:
            [batch_size, node_dim + global_dim] if global_features provided
            [batch_size, node_dim] otherwise
        """
        # Pool node features
        pooled = self.pool_fn(node_features, batch_idx)

        # Concatenate with global features if provided
        if global_features is not None:
            return torch.cat([pooled, global_features], dim=-1)
        return pooled


class AttentiveFPReadoutAndConcatGlobal(nn.Module):
    """
    AttentiveFP readout with global feature concatenation.
    Mimics TensorFlow's AttentiveFPReadoutAndConcatGlobal.
    """

    def __init__(self, node_dim: int, steps: int = 4, **kwargs):
        super().__init__()
        self.readout = AttentiveFPReadout(node_dim=node_dim, steps=steps)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        batch_idx: torch.Tensor,
        global_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            node_features: [num_nodes, node_dim]
            edge_index: [2, num_edges]
            batch_idx: [num_nodes] batch assignment
            global_features: [batch_size, global_dim] or None

        Returns:
            [batch_size, node_dim + global_dim] if global_features provided
            [batch_size, node_dim] otherwise
        """
        # Apply AttentiveFP readout
        pooled = self.readout(node_features, edge_index, batch_idx)

        # Concatenate with global features if provided
        if global_features is not None:
            return torch.cat([pooled, global_features], dim=-1)
        return pooled
