import sys
import os
import unittest
from unittest.mock import MagicMock, patch

# Add root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock modules to allow import
sys.modules['torch'] = MagicMock()
sys.modules['torch.nn'] = MagicMock()
sys.modules['numpy'] = MagicMock()
sys.modules['step1_preprocessing'] = MagicMock()
sys.modules['step3_gkan_model'] = MagicMock()
sys.modules['step2_kan_layer'] = MagicMock()
sys.modules['torch_geometric'] = MagicMock()
sys.modules['torch_geometric.loader'] = MagicMock()

import torch
import numpy as np

class TestOptimization(unittest.TestCase):
    def test_expand_instead_of_repeat(self):
        # Reload to pick up mocks
        if 'step4_train_interpret' in sys.modules:
            del sys.modules['step4_train_interpret']
        import step4_train_interpret

        kan_layer = MagicMock()
        kan_layer.in_dim = 5
        kan_layer.out_dim = 3
        kan_layer.grid = MagicMock()
        kan_layer.coef = MagicMock()
        kan_layer.k = 3
        kan_layer.base_fun = MagicMock()
        kan_layer.scale_base = MagicMock()
        kan_layer.scale_sp = MagicMock()

        # Set up a chain of mocks for the tensor operations
        mock_linspace_result = MagicMock()
        torch.linspace.return_value = mock_linspace_result

        mock_unsqueeze_result = MagicMock()
        mock_linspace_result.unsqueeze.return_value = mock_unsqueeze_result

        mock_expand_result = MagicMock()
        mock_unsqueeze_result.expand.return_value = mock_expand_result

        # Configure numpy mocks to avoid errors during formula extraction
        # y_np = y_total.cpu().numpy()
        mock_y_total = MagicMock()
        # In get_symbolic_formula: y_total = y_eval * ... + y_base
        # We need to make sure the math doesn't crash the mock

        with patch('step4_train_interpret.coef2curve') as mock_coef2curve:
            mock_coef2curve.return_value = MagicMock()

            # Run the function - it should call expand
            try:
                step4_train_interpret.get_symbolic_formula(kan_layer, device='cpu')
            except Exception:
                # We expect it might still fail later due to heavy mocking,
                # but the call to expand should happen early.
                pass

        # Assertions on implementation details (as requested by the task to ensure .expand is used)

        # 1. Verify linspace called with device
        torch.linspace.assert_called()
        _, kwargs = torch.linspace.call_args
        self.assertEqual(kwargs.get('device'), 'cpu')

        # 2. Verify expand called instead of repeat
        mock_unsqueeze_result.expand.assert_called_once_with(-1, kan_layer.in_dim)
        self.assertFalse(mock_unsqueeze_result.repeat.called, "repeat() should not be called")

if __name__ == '__main__':
    unittest.main()
