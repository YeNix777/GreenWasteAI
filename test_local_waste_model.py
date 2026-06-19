import unittest

from PIL import Image

from local_waste_model import analyze_with_local_model, image_feature_vector
from wastewise import disposal_advice


class LocalWasteModelTests(unittest.TestCase):
    def test_feature_vector_has_expected_length(self):
        image = Image.new("RGB", (16, 16), color=(255, 0, 0))
        import io

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")

        self.assertEqual(len(image_feature_vector(buffer.getvalue())), 57)

    def test_local_prediction_maps_to_disposal_advice(self):
        model = {
            "centroids": {
                "organic": [1.0] + [0.0] * 56,
                "battery": [0.0, 1.0] + [0.0] * 55,
            }
        }
        image = Image.new("RGB", (16, 16), color=(0, 0, 0))
        import io

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")

        recognition = analyze_with_local_model(buffer.getvalue(), model)
        self.assertIn(recognition.category, {"organic", "battery"})
        self.assertTrue(disposal_advice(recognition).bin_name)


if __name__ == "__main__":
    unittest.main()
