import unittest

from omniintents.agentgpt.agentgpt import AgentGPT
from omniintents.llm import MockLLMClient
from omniintents.types import AgentType, Capability, TaskPlan, TaskStep


class TestAgentGPT(unittest.TestCase):
    def test_voice_only(self):
        llm = MockLLMClient()
        gpt = AgentGPT(llm=llm)
        plan = TaskPlan(goal="Talk", steps=[TaskStep(1, "Say hello", [Capability.SPEECH_IO])])
        rec = gpt.recommend(plan)
        self.assertEqual(rec.agent_type, AgentType.VOICE_ASSISTANT)

    def test_text_io_requires_digital(self):
        llm = MockLLMClient()
        gpt = AgentGPT(llm=llm)
        plan = TaskPlan(goal="Search", steps=[TaskStep(1, "Search web", [Capability.TEXT_IO, Capability.SPEECH_IO])])
        rec = gpt.recommend(plan)
        self.assertEqual(rec.agent_type, AgentType.DIGITAL_ASSISTANT)


if __name__ == "__main__":
    unittest.main()
