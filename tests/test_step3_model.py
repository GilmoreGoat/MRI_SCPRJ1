import sys
import unittest
from unittest.mock import MagicMock

# Create basic mocks
mock_torch = MagicMock()
mock_torch_geometric = MagicMock()
mock_numpy = MagicMock()
mock_networkx = MagicMock()

# Define a MockModule that behaves like nn.Module
class MockModule:
    def __init__(self, *args, **kwargs):
        pass

    def register_buffer(self, name, tensor):
        setattr(self, name, tensor)

    def to(self, device):
        return self

    def __call__(self, *args, **kwargs):
        if hasattr(self, 'forward'):
            return self.forward(*args, **kwargs)
        return MagicMock()

# Define MockModuleList
class MockModuleList(list):
    def __init__(self, modules=None):
        super().__init__()
        if modules:
            self.extend(modules)

# Setup torch mocks
mock_torch.nn.Module = MockModule
mock_torch.nn.ModuleList = MockModuleList
mock_torch.nn.Parameter = MagicMock(return_value=MagicMock())
mock_torch.Tensor = MagicMock
mock_torch.long = MagicMock()
mock_torch.float = MagicMock()
mock_torch.linspace = MagicMock(return_value=MagicMock())
mock_torch.cat = MagicMock(return_value=MagicMock())
mock_torch.zeros = MagicMock(return_value=MagicMock())
mock_torch.rand = MagicMock(return_value=MagicMock())
mock_torch.randn = MagicMock(return_value=MagicMock())
mock_torch.einsum = MagicMock(return_value=MagicMock())
mock_torch.sum = MagicMock(return_value=MagicMock())
mock_torch.nan_to_num = MagicMock(return_value=MagicMock())
mock_torch.linalg.lstsq = MagicMock(return_value=MagicMock(solution=MagicMock()))

# Setup torch_geometric mocks
# MessagePassing should inherit from MockModule so GKANConv works
class MockMessagePassing(MockModule):
    def __init__(self, aggr=None):
        super().__init__()
        self.aggr = aggr

    def propagate(self, edge_index, **kwargs):
        # return a mock tensor
        return MagicMock()

mock_torch_geometric.nn.MessagePassing = MockMessagePassing
mock_torch_geometric.nn.global_mean_pool = MagicMock()

# Configure add_self_loops to return an iterable edge_index
mock_edge_index = MagicMock()
mock_edge_index.__iter__.side_effect = lambda: iter([MagicMock(), MagicMock()])
mock_torch_geometric.utils.add_self_loops = MagicMock(return_value=(mock_edge_index, MagicMock()))

mock_torch_geometric.utils.degree = MagicMock(return_value=MagicMock())
mock_torch_geometric.utils.from_networkx = MagicMock()
mock_torch_geometric.data.Data = MagicMock()

# Apply to sys.modules
sys.modules['torch'] = mock_torch
sys.modules['torch.nn'] = mock_torch.nn
sys.modules['torch.linalg'] = mock_torch.linalg
sys.modules['torch_geometric'] = mock_torch_geometric
sys.modules['torch_geometric.nn'] = mock_torch_geometric.nn
sys.modules['torch_geometric.utils'] = mock_torch_geometric.utils
sys.modules['torch_geometric.data'] = mock_torch_geometric.data
sys.modules['numpy'] = mock_numpy
sys.modules['networkx'] = mock_networkx

# Import modules under test
import step2_kan_layer
# Mock internal functions of step2_kan_layer to avoid complex tensor logic on Mocks
step2_kan_layer.curve2coef = MagicMock(return_value=MagicMock())
step2_kan_layer.coef2curve = MagicMock(return_value=MagicMock())
step2_kan_layer.B_batch = MagicMock(return_value=MagicMock())

from step3_gkan_model import GKAN, GKANConv

class TestGKAN(unittest.TestCase):
    def test_gkan_structure(self):
        print("Testing GKAN structure with custom mocks...")

        in_channels = 10
        hidden_channels = 16
        out_channels = 3

        model = GKAN(in_channels, hidden_channels, out_channels, num_layers=2)

        # Check layers
        self.assertEqual(len(model.convs), 2)
        self.assertIsInstance(model.convs[0], GKANConv)
        self.assertIsInstance(model.head, step2_kan_layer.KANLinear)
        print("Structure check passed.")

    def test_gkan_forward(self):
        print("Testing GKAN forward pass...")

        in_channels = 10
        hidden_channels = 16
        out_channels = 3

        model = GKAN(in_channels, hidden_channels, out_channels)

        mock_data = MagicMock()
        mock_data.x = MagicMock()
        mock_data.edge_index = MagicMock()
        mock_data.batch = MagicMock()

        print(f"Model pool: {model.pool}")
        print(f"Mock pool: {mock_torch_geometric.nn.global_mean_pool}")

        out = model(mock_data)

        # Check global_mean_pool called
        mock_torch_geometric.nn.global_mean_pool.assert_called()
        print("Forward pass check passed.")

if __name__ == "__main__":
    unittest.main()
