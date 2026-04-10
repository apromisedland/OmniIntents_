import unittest

from omniintents.intentgpt.credibility import CredibilityBasedAttentionShifter


class TestCredibility(unittest.TestCase):
    def test_brightness_threshold(self):
        shifter = CredibilityBasedAttentionShifter(brightness_credible_threshold=0.15, speech_confidence_credible_threshold=0.70)
        visual = {"brightness": {"value": 0.10}}
        audio = {"speech": {"transcript": "hi", "confidence": 0.90}}
        report = shifter.evaluate(visual, audio)
        self.assertFalse(report["visual"]["credible"])
        self.assertTrue(report["voice"]["credible"])

    def test_missing_modalities(self):
        shifter = CredibilityBasedAttentionShifter(missing_visual_is_incredible=True, missing_audio_is_incredible=True)
        visual = {}
        audio = {}
        report = shifter.evaluate(visual, audio)
        self.assertFalse(report["visual"]["credible"])
        self.assertFalse(report["voice"]["credible"])


if __name__ == "__main__":
    unittest.main()
