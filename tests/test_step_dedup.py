import unittest

from d3study.step_dedup import drop_same_observations, drop_text_near_duplicates, should_verify


class StepDedupTests(unittest.TestCase):
    def test_jaccard_drops_near_duplicate_actions(self):
        commands = [
            "sed -n 1,80p pydicom/valuerep.py",
            "sed -n 1,80p pydicom/valuerep.py",
            "python -m pytest pydicom/tests/test_filereader.py::TestReader::test_lut_descriptor",
        ]
        self.assertEqual(drop_text_near_duplicates(commands), [0, 2])

    def test_jaccard_keeps_actions_below_the_threshold(self):
        # 4 shared tokens out of 6 is below 0.8.
        commands = ["alpha beta gamma delta epsilon", "alpha beta gamma delta zeta"]
        self.assertEqual(drop_text_near_duplicates(commands), [0, 1])

    def test_identical_nonempty_observations_collapse(self):
        outputs = ["LUTDescriptor: [-2816, -2048, 16]\n", "", "LUTDescriptor: [-2816, -2048, 16]\n", "different\n"]
        self.assertEqual(drop_same_observations(outputs), [0, 1, 3])

    def test_verification_requires_two_unique_candidates(self):
        self.assertFalse(should_verify(0))
        self.assertFalse(should_verify(1))
        self.assertTrue(should_verify(2))
