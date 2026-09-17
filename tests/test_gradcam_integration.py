"""Integration with repository weights, not evidence of disease localization."""
import unittest

import keras
import numpy as np

from pathlib import Path

from model_utils import (
    MODEL_PATHS, GRADCAM_LAYERS, generate_gradcam_heatmap, preprocess_image,
)


class GournetIntegrationTests(unittest.TestCase):
    def test_saved_v2_gradcam_executes_without_changing_predictions(self):
        model = keras.models.load_model(str(MODEL_PATHS["v2"]), compile=False)
        # No original leaf files were supplied locally. Use a deterministic RGB
        # fixture only to verify runtime integration, never localization quality.
        image = np.random.default_rng(123).uniform(0, 255, (1, 224, 224, 3)).astype("float32")
        before = model(image, training=False).numpy()
        predicted_class = int(before[0].argmax())
        heatmap = generate_gradcam_heatmap(model, GRADCAM_LAYERS["v2"], image, predicted_class)
        self.assertEqual(heatmap.shape, (28, 28))
        self.assertTrue(np.isfinite(heatmap).all())
        self.assertGreater(float(heatmap.max()), 0.0)
        self.assertGreaterEqual(float(heatmap.min()), 0.0)
        self.assertLessEqual(float(heatmap.max()), 1.0)
        np.testing.assert_array_equal(model(image, training=False).numpy(), before)


if __name__ == "__main__":
    unittest.main()
