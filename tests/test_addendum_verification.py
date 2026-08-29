import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


def load_fixture(name):
    path = ROOT / "verification" / f"gate_{name}.py"
    spec = importlib.util.spec_from_file_location(f"gate_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class AddendumVerificationFixtureTests(unittest.TestCase):
    def test_gate_A_kinematic_fixture(self):
        result = load_fixture("A_bertin_rotation").run_fixture()
        self.assertTrue(result["fixture_passed"])
        self.assertFalse(result["scientific_gate_passed"])

    def test_gate_B_signed_patterning_fixture(self):
        result = load_fixture("B_signed_patterning").run_fixture()
        self.assertTrue(result["fixture_passed"])
        self.assertFalse(result["scientific_gate_passed"])

    def test_gate_C_polygonization_fixture(self):
        result = load_fixture("C_polygonization").run_fixture()
        self.assertTrue(result["fixture_passed"])
        self.assertFalse(result["scientific_gate_passed"])

    def test_gate_D_frank_bilby_fixture(self):
        result = load_fixture("D_frank_bilby_rotation").run_fixture()
        self.assertTrue(result["fixture_passed"])
        self.assertFalse(result["scientific_gate_passed"])


if __name__ == "__main__":
    unittest.main()
