# ----------------------------------------------------------------------------------------------------
# IBM Confidential
# Licensed Materials - Property of IBM
# 5737-H76, 5900-A3Q
# © Copyright IBM Corp. 2025  All Rights Reserved.
# US Government Users Restricted Rights - Use, duplication or disclosure restricted by
# GSA ADPSchedule Contract with IBM Corp.
# ----------------------------------------------------------------------------------------------------

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class EvaluationCriterion:
    name: str = None
    description: str = None

    def __post_init__(self):
        if not isinstance(self.name, str):
            raise TypeError(
                f"name must be str, got {type(self.name).__name__}")
        if not isinstance(self.description, str):
            raise TypeError(
                f"description must be str, got {type(self.description).__name__}")

    @classmethod
    def from_dict(cls, criterion_dict):
        if "name" not in criterion_dict:
            raise ValueError(
                f"criterion_dict must contain key 'name', got {criterion_dict}")
        if "description" not in criterion_dict:
            raise ValueError(
                f"criterion_dict must contain key 'description', got {criterion_dict}")
        return cls(criterion_dict.get("name"), criterion_dict.get("description"))

    def to_str(self):
        return f"{self.name}: {self.description}"

    def to_dict(self):
        return {"name": self.name, "description": self.description}


@dataclass
class EvaluationCriteria:
    criteria_list: List[EvaluationCriterion]

    @classmethod
    def from_dict(cls, criteria_dict: Dict[str, str]):
        """ create from dictionary: {"name":"value"} """
        criteria_list = [EvaluationCriterion.from_dict({"name": name, "description": description})
                         for name, description in criteria_dict.items()]
        return cls(criteria_list)

    @classmethod
    def from_list_of_dicts(cls, criteria_dict: List[Dict[str, str]]):
        """ create from dictionary. each dict should be of format: {"name":"...", "description":""}"""
        criteria_list = [EvaluationCriterion.from_dict(
            d) for d in criteria_dict]
        return cls(criteria_list)

    def to_str(self):
        return "\n".join(c.to_str() for c in self.criteria_list)

    def to_list_of_dicts(self):
        return [c.to_dict() for c in self.criteria_list]

    def to_dict(self):
        return {c.name: c.description for c in self.criteria_list}


def get_default_evaluation_criteria(agent_mode: bool) -> EvaluationCriteria:
    eval_dict = default_agentic_eval_criteria_dict if agent_mode else default_eval_criteria_dict
    return EvaluationCriteria.from_dict(eval_dict)


default_eval_criteria_dict = {
    "Adherence to Instructions and Relevance":
        "Does the model follow the given instructions (if any) and provide a relevant response to the input?",
    "Accuracy & Completeness":
        "Is the response factually correct (if applicable) and does it fully address the input's request without "
        "omitting critical details?",
    "Coherence & Clarity":
        "Does the response make sense, follow a logical flow, and is it easy to understand?",
}

default_agentic_eval_criteria_dict = {
    "Correctness": "The output is factually correct and logically sound for what THIS STEP is supposed to do. For a tool call: correct tool selected with valid parameters. For a reasoning step: sound logic that advances the task. For a final answer: meets the user's requirements.",
    "Completeness": "The step fulfills its specific role in the workflow. A tool call is complete if it invokes the right tool with all required parameters. A reasoning step is complete if it reaches a conclusion. Only final answers need to fully address the user's request.",
    "Relevance": "The output is appropriate for this step's role — it advances the task without unnecessary or off-topic content. A tool call to gather needed information is relevant even if it doesn't answer the user directly.",
    "Tool Selection": "If tools were called: were the appropriate tools chosen given the available options? Were the arguments well-formed and suitable for the task? If no tools were called but should have been, note this. Skip this criterion if no tools are involved.",
}
