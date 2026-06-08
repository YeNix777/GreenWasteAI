import unittest

from streamlit.testing.v1 import AppTest


class TrennklarAppTests(unittest.TestCase):
    def test_app_loads_and_demo_produces_advice(self):
        app = AppTest.from_file("app.py", default_timeout=20).run()

        self.assertEqual(len(app.exception), 0)
        rendered = "\n".join(element.value for element in app.markdown)
        self.assertIn("Trennklar", rendered)

        app.segmented_control[0].select("Demo ausprobieren")
        app.run()
        app.selectbox[0].select("Batterie")
        app.button[0].click()
        app.run()

        self.assertEqual(len(app.exception), 0)
        rendered = "\n".join(element.value for element in app.markdown)
        self.assertIn("Batteriesammlung", rendered)
        self.assertIn("Haushaltsbatterie", rendered)


if __name__ == "__main__":
    unittest.main()
