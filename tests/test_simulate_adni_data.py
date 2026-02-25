
import unittest
import numpy as np
from step1_preprocessing import simulate_adni_data

class TestSimulateAdniData(unittest.TestCase):
    def test_simulate_adni_data_shape(self):
        num_patients = 10
        num_rois = 90
        features, labels = simulate_adni_data(num_patients, num_rois)
        self.assertEqual(features.shape, (num_patients, num_rois, 2))
        self.assertEqual(labels.shape, (num_patients,))

    def test_simulate_adni_data_values_range(self):
        num_patients = 100
        num_rois = 90
        features, labels = simulate_adni_data(num_patients, num_rois)
        # Check normalization range [-1, 1]
        self.assertTrue(np.all(features >= -1.0))
        self.assertTrue(np.all(features <= 1.0))

    def test_simulate_adni_data_deterministic(self):
        num_patients = 10
        num_rois = 90
        seed = 42
        features1, labels1 = simulate_adni_data(num_patients, num_rois, seed)
        features2, labels2 = simulate_adni_data(num_patients, num_rois, seed)
        np.testing.assert_array_equal(features1, features2)
        np.testing.assert_array_equal(labels1, labels2)

    def test_simulate_adni_data_logic(self):
        """Verify that AD and MCI patients have reduced values in sensitive ROIs."""
        num_patients = 100
        num_rois = 90
        seed = 42
        # Use a fixed seed to ensure we get some AD and MCI patients
        features, labels = simulate_adni_data(num_patients, num_rois, seed)

        sensitive_rois = list(range(10))

        # We can't easily check exact values because of random noise and normalization.
        # But we can check if the mean of AD is lower than Normal in sensitive ROIs.
        # Or better, we can manually apply the logic on a copy and check if it matches.

        # Re-implement logic for verification
        np.random.seed(seed)
        labels_check = np.random.choice([0, 1, 2], size=num_patients)
        features_base = np.zeros((num_patients, num_rois, 2))
        features_base[:, :, 0] = np.random.normal(2.5, 0.3, (num_patients, num_rois))
        features_base[:, :, 1] = np.random.normal(2000, 500, (num_patients, num_rois))

        for i in range(num_patients):
            if labels_check[i] == 2: # AD
                features_base[i, sensitive_rois, 0] -= 0.5
                features_base[i, sensitive_rois, 1] -= 800
            elif labels_check[i] == 1: # MCI
                features_base[i, sensitive_rois, 0] -= 0.2
                features_base[i, sensitive_rois, 1] -= 300

        # Normalize
        for i in range(2):
            feat = features_base[:, :, i]
            min_val = feat.min()
            max_val = feat.max()
            if max_val > min_val:
                features_base[:, :, i] = 2 * (feat - min_val) / (max_val - min_val) - 1
            else:
                features_base[:, :, i] = 0.0

        np.testing.assert_array_almost_equal(features, features_base, decimal=5)

if __name__ == '__main__':
    unittest.main()
