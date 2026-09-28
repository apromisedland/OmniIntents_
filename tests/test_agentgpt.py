import pytest

from omniintents import OmniIntentsConfig
from omniintents.agentgpt.agentgpt import AgentGPT
from omniintents.llm import MockLLMClient
from omniintents.types import AgentType, Capability, TaskPlan, TaskStep
from conftest import CaptureClient


@pytest.mark.parametrize("strategy", ["implicit", "explicit"])
@pytest.mark.parametrize("capabilities,expected", [
    ([Capability.SPEECH_IO, Capability.TEXT_IO], AgentType.VOICE_ASSISTANT),
    ([Capability.SPEECH_IO, Capability.NAVIGATION_GUIDANCE], AgentType.AR_AGENT),
    ([Capability.SPEECH_IO, Capability.CREATIVITY], AgentType.GENERATIVE_AI_AGENT),
    ([Capability.PHYSICAL_MOVEMENT], AgentType.PHYSICAL_AGENT),
])
def test_agent_semantics(strategy, capabilities, expected):
    plan = TaskPlan("example", [TaskStep(1, "example", capabilities)])
    result = AgentGPT(MockLLMClient(), OmniIntentsConfig(agent_strategy=strategy)).recommend(plan)
    assert result.agent_type == expected
    assert result.missing_capabilities == []


def test_valid_llm_choice_not_overwritten_by_cheapest():
    client = CaptureClient({"agent": {"agent_type": "ar_agent", "rationale": "Choose visual embodiment."}})
    plan = TaskPlan("Talk", [TaskStep(1, "Talk", [Capability.SPEECH_IO])])
    result = AgentGPT(client).recommend(plan)
    assert result.agent_type == AgentType.AR_AGENT


def test_missing_capability_is_not_silent_success():
    client = CaptureClient({"agent": {"agent_type": "voice_assistant", "rationale": "Incorrect model choice."}})
    plan = TaskPlan("Carry an item", [TaskStep(1, "Carry", [Capability.PHYSICAL_INTERACTION])])
    result = AgentGPT(client).recommend(plan)
    assert result.agent_type is None
    assert result.status == "no_suitable_agent"
    assert result.missing_capabilities == [Capability.PHYSICAL_INTERACTION]


def test_no_agent_has_combined_creativity_and_manipulation():
    plan = TaskPlan("Mixed", [TaskStep(1, "Mixed", [Capability.CREATIVITY, Capability.PHYSICAL_INTERACTION])])
    result = AgentGPT(MockLLMClient()).recommend(plan)
    assert result.agent_type is None


def test_mock_stage_does_not_depend_on_prompt_markers():
    from omniintents.llm.base import LLMRequest
    from omniintents.utils.json_utils import strict_loads

    request = LLMRequest("agent_flags", "TASK_PLAN_JSON AGENT_RECOMMENDATION_JSON", {
        "task_plan": TaskPlan("Hello", [TaskStep(1, "Speak", [Capability.SPEECH_IO])]).to_json_dict(),
    })
    response = strict_loads(MockLLMClient().generate(request).text)
    assert response["speech_expression"] is True
    assert "steps" not in response
