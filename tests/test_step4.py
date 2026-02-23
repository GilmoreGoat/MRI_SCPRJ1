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
sys.modules['torch_geometric.data'] = MagicMock()
sys.modules['torch_geometric.utils'] = MagicMock()
sys.modules['networkx'] = MagicMock()
sys.modules['numpy'] = MagicMock()

# Mock submodules to avoid import errors
sys.modules['step1_preprocessing'] = MagicMock()
sys.modules['step3_gkan_model'] = MagicMock()
sys.modules['step2_kan_layer'] = MagicMock()

class TestStep4(unittest.TestCase):
    def test_train_model(self):
        if 'step4_train_interpret' in sys.modules:
            del sys.modules['step4_train_interpret']
        import step4_train_interpret

        # Setup mocks
        model = MagicMock()
        # Train loader iterates twice
        train_loader = [MagicMock(), MagicMock()]
        # Val loader iterates once
        val_loader = [MagicMock()]
        optimizer = MagicMock()
        criterion = MagicMock()

        # Configure data.to(device) to return data itself to preserve mocks
        for data in train_loader + val_loader:
            data.to.return_value = data

        # Configure loss.item() to return a float
        criterion.return_value.item.return_value = 0.5

        # Configure model output
        outputs = MagicMock()
        outputs.data = MagicMock()
        model.return_value = outputs

        # Configure torch.max to return (values, predicted)
        predicted_mock = MagicMock()
        step4_train_interpret.torch.max.return_value = (None, predicted_mock)

        # Configure predicted.eq().sum().item()
        predicted_mock.eq.return_value.sum.return_value.item.return_value = 1

        # Configure data.y.size(0)
        for data in val_loader:
             data.y.size.return_value = 2 # batch size 2

        # Run
        train_losses, val_accuracies = step4_train_interpret.train_model(
            model, train_loader, val_loader, optimizer, criterion, epochs=1
        )

        # Assertions
        self.assertTrue(model.train.called)
        self.assertTrue(model.eval.called)
        self.assertEqual(optimizer.step.call_count, 2)
        self.assertEqual(len(train_losses), 1)
        self.assertEqual(len(val_accuracies), 1)
        self.assertIsInstance(train_losses[0], float)
        self.assertIsInstance(val_accuracies[0], float)

    def test_interpret_kan_flow(self):
        if 'step4_train_interpret' in sys.modules:
            del sys.modules['step4_train_interpret']
        import step4_train_interpret

        model = MagicMock()
        data = MagicMock()

        # Mock convs structure: model.convs[0].kan
        kan_layer = MagicMock()
        conv_layer = MagicMock()
        conv_layer.kan = kan_layer
        model.convs = [conv_layer]

        # Mock helper functions
        with patch('step4_train_interpret.get_symbolic_formula') as mock_get_formula:
             mock_get_formula.return_value = {"Feature_0->Hidden_0": "0.5*x + 0.1"}

             with patch('step4_train_interpret.get_top_connections') as mock_get_top:
                 mock_get_top.return_value = [(0, 0.9), (1, 0.8)]

                 formulas, top_regions = step4_train_interpret.interpret_kan(model, data)

                 # Verify calls
                 mock_get_formula.assert_called_once()
                 mock_get_top.assert_called_once()

                 # Verify output
                 self.assertEqual(len(formulas), 1)
                 self.assertEqual(len(top_regions), 2)
                 self.assertEqual(top_regions[0], (0, 0.9))

if __name__ == '__main__':
    unittest.main()
