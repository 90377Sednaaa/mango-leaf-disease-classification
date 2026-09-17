"""Behavioral Grad-CAM tests; no leaf-localization claims from synthetic inputs."""
import unittest

import keras
import numpy as np
from PIL import Image

from model_utils import GRADCAM_LAYERS, generate_gradcam_heatmap, overlay_gradcam


def make_model():
    inputs = keras.Input((2, 2, 1))
    features = keras.layers.Conv2D(1, 1, use_bias=False, name="features")(inputs)
    pooled = keras.layers.GlobalAveragePooling2D()(features)
    outputs = keras.layers.Dense(2, activation="softmax", name="classifier")(pooled)
    model = keras.Model(inputs, outputs)
    model.get_layer("features").set_weights([np.ones((1, 1, 1, 1), np.float32)])
    # Class 0 has a positive logit gradient, but a NEGATIVE probability gradient:
    # its competitor's score increases twice as fast. The bias saturates softmax.
    model.get_layer("classifier").set_weights([
        np.array([[1.0, 2.0]], np.float32), np.array([1000.0, 0.0], np.float32)
    ])
    return model


class GradcamTests(unittest.TestCase):
    def setUp(self):
        self.model = make_model()
        self.image = np.array([[[[1.0], [2.0]], [[3.0], [4.0]]]], np.float32)

    def test_v2_targets_post_normalization_features(self):
        self.assertEqual(GRADCAM_LAYERS["v2"], "Block4_ReLU")

    def test_saturated_softmax_still_explains_positive_class_logit(self):
        heatmap = generate_gradcam_heatmap(self.model, "features", self.image)
        np.testing.assert_allclose(heatmap, [[0.25, 0.5], [0.75, 1.0]], atol=1e-6)

    def test_explanation_does_not_mutate_predictions_or_weights(self):
        before = self.model(self.image, training=False).numpy()
        weights = [w.copy() for w in self.model.get_weights()]
        activation = self.model.layers[-1].activation
        generate_gradcam_heatmap(self.model, "features", self.image, pred_index=1)
        np.testing.assert_array_equal(self.model(self.image, training=False).numpy(), before)
        self.assertIs(self.model.layers[-1].activation, activation)
        for before_weight, after_weight in zip(weights, self.model.get_weights()):
            np.testing.assert_array_equal(before_weight, after_weight)

    def test_explicit_class_uses_requested_score(self):
        self.model.layers[-1].set_weights([
            np.array([[1.0, -2.0]], np.float32), np.zeros(2, np.float32)
        ])
        heatmap = generate_gradcam_heatmap(self.model, "features", self.image, pred_index=1)
        np.testing.assert_array_equal(heatmap, np.zeros((2, 2)))

    def test_empty_heatmap_does_not_tint_original_blue(self):
        original = Image.new("RGB", (8, 6), (100, 150, 200))
        result = overlay_gradcam(original, np.zeros((2, 2), np.float32))
        np.testing.assert_array_equal(np.asarray(result), np.asarray(original))

    def test_rejects_multi_image_batch_instead_of_averaging_explanations(self):
        with self.assertRaisesRegex(ValueError, "single image"):
            generate_gradcam_heatmap(self.model, "features", np.repeat(self.image, 2, axis=0))

    def test_rejects_invalid_class_index(self):
        with self.assertRaisesRegex(ValueError, "pred_index"):
            generate_gradcam_heatmap(self.model, "features", self.image, pred_index=-1)


if __name__ == "__main__":
    unittest.main()
