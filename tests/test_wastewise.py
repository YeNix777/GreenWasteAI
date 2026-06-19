import unittest

from wastewise import Recognition, disposal_advice, recognition_from_dict


class WasteWiseTests(unittest.TestCase):
    def test_packaging_goes_to_duisburg_recycling_bin(self):
        recognition = Recognition(
            "Joghurtbecher", "Kunststoff", "packaging", "leer", False, 0.9, ""
        )
        self.assertEqual(disposal_advice(recognition).bin_name, "Wertstofftonne")

    def test_battery_overrides_normal_household_waste(self):
        recognition = Recognition(
            "Akku", "Metall", "battery", "gebraucht", True, 0.98, ""
        )
        advice = disposal_advice(recognition)
        self.assertEqual(advice.bin_name, "Batteriesammlung")
        self.assertIn("Brandgefahr", advice.warning)

    def test_unknown_confidence_is_clamped(self):
        recognition = recognition_from_dict(
            {"item": "Etwas", "category": "unknown", "confidence": 4}
        )
        self.assertEqual(recognition.confidence, 1.0)


if __name__ == "__main__":
    unittest.main()
