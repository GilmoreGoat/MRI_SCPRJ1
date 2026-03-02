import sys
import os
import unittest
from unittest.mock import MagicMock, patch

# Add root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock torch, numpy, etc. before importing anything else
sys.modules['torch'] = MagicMock()
sys.modules['torch.nn'] = MagicMock()
sys.modules['torch.optim'] = MagicMock()
sys.modules['torch.utils.data'] = MagicMock()
sys.modules['torch_geometric'] = MagicMock()
sys.modules['torch_geometric.loader'] = MagicMock()
sys.modules['torch_geometric.data'] = MagicMock()
sys.modules['torch_geometric.utils'] = MagicMock()
sys.modules['networkx'] = MagicMock()
sys.modules['numpy'] = MagicMock()

# Mock submodules to avoid import errors
sys.modules['step1_preprocessing'] = MagicMock()
sys.modules['step3_gkan_model'] = MagicMock()
sys.modules['step2_kan_layer'] = MagicMock()

class TestSecurityFix(unittest.TestCase):
    def test_train_model_zero_val_loader(self):
        if 'step4_train_interpret' in sys.modules:
            del sys.modules['step4_train_interpret']
        import step4_train_interpret

        # Setup mocks
        model = MagicMock()
        train_loader = [MagicMock()]
        val_loader = [] # Empty val_loader to trigger total = 0
        optimizer = MagicMock()
        criterion = MagicMock()

        # Configure data.to(device) to return data itself to preserve mocks
        for data in train_loader:
            data.to.return_value = data

        # Configure loss.item() to return a float
        criterion.return_value.item.return_value = 0.5

        # Run
        try:
            train_losses, val_accuracies = step4_train_interpret.train_model(
                model, train_loader, val_loader, optimizer, criterion, epochs=1
            )
            # If it reaches here, no ZeroDivisionError occurred
            self.assertEqual(len(val_accuracies), 1)
            self.assertEqual(val_accuracies[0], 0.0)
        except ZeroDivisionError:
            self.fail("train_model raised ZeroDivisionError with empty val_loader!")

if __name__ == '__main__':
    unittest.main()
