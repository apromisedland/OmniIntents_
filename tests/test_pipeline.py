import unittest

from omniintents import MultimodalInput, OmniIntentsPipeline
from omniintents.llm import MockLLMClient
from omniintents.types import AgentType


class TestPipeline(unittest.TestCase):
    def test_end_to_end_text_only(self):
        llm = MockLLMClient()
        pipe = OmniIntentsPipeline(llm=llm)
        mm = MultimodalInput(utterance="query information about this book")
        result, trace = pipe.run_with_trace(mm)
        self.assertTrue(result.intent.label)
        self.assertTrue(result.task_plan.goal)
        self.assertTrue(result.agent.agent_type.value)
        # Trace should contain some events
        self.assertGreater(len(trace.events), 2)

    def test_search_intent_and_agent(self):
        llm = MockLLMClient()
        pipe = OmniIntentsPipeline(llm=llm)
        mm = MultimodalInput(
            utterance="query information about this book",
            objects=["book"],
            context_location="library",
            speech_transcript="Query information about this book.",
            speech_confidence=0.95,
        )
        res = pipe.run(mm)
        self.assertIn("Request Search", res.intent.label)
        self.assertEqual(res.agent.agent_type, AgentType.MULTIMODAL_ASSISTANT)

    def test_physical_agent(self):
        llm = MockLLMClient()
        pipe = OmniIntentsPipeline(llm=llm)
        mm = MultimodalInput(utterance="cut vegetables")
        res = pipe.run(mm)
        self.assertEqual(res.agent.agent_type, AgentType.PHYSICAL_ROBOT)


if __name__ == "__main__":
    unittest.main()
