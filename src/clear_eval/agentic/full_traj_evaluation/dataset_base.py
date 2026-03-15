"""
Dataset Base Class and Implementations
========================================

Defines an abstract base class for dataset-specific logic and concrete
implementations for CUGA, FinOps, and WXO datasets.

To add a new dataset:
1. Create a class inheriting from DatasetBase
2. Implement the 3 required methods
3. Register in DATASET_REGISTRY
"""
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Type

DEFAULT_MODEL_CONTEXT_TOKENS = 128,000
TRAJ_DATA_DIR = Path(__file__).parent.parent/"data/paper_experiments" # "data/traj_full_data"
RESULTS_DIR  = Path(__file__).parent.parent/"output/paper_experiments/full_traj"   #"output/full_traj"


# =============================================================================
# Results Directory Management
# =============================================================================

def get_results_dir(base_name: str, model_id: str, results_base: Path = Path(RESULTS_DIR)) -> Path:
    """
    Get standardized results directory path.

    Args:
        base_name: Base directory name (e.g., "final_answer_results", "success_results")
        model_id: Model identifier (will be sanitized)
        results_base: Base results directory path

    Returns:
        Path to results directory
    """
    safe_name = model_id.replace("/", "_").replace(":", "_").replace("-", "_")
    return RESULTS_DIR / results_base / f"{base_name}/{safe_name}"


class DatasetBase(ABC):
    """Abstract base class for dataset-specific implementations."""
    
    def __init__(self, data_dir: Path = None):
        """
        Initialize dataset.
        
        Args:
            data_dir: Optional override for data directory path
        """
        self.data_dir = data_dir or self.get_default_data_dir()
        self.name = ""


    def get_default_data_dir(self) -> Path:
        """Return the default data directory for this dataset."""
        pass

    def format_trajectory(self, traj_data: dict) -> str:
        """
        Format trajectory JSON into human-readable text for judge model.
        
        Args:
            traj_data: The trajectory JSON data loaded from file
            
        Returns:
            Human-readable string representation of the trajectory
        """
        return json.dumps(traj_data)


    def extract_final_response(self, traj_data: dict) -> str:
        """
        Extract the agent's final answer/output from the trajectory.
        
        Args:
            traj_data: The trajectory JSON data loaded from file
            
        Returns:
            The agent's final answer/output as a string
        """
        steps = traj_data.get("steps", [])
        if steps:
            return steps[-1].get("content")
        return None


    def extract_user_request(self, traj_data: dict) -> str:
        return traj_data.get("intent")


# =============================================================================
# CUGA Dataset Implementation
# =============================================================================

class CUGADataset(DatasetBase):
    """CUGA dataset implementation."""

    name = "CUGA"
    
    def get_default_data_dir(self) -> Path:
        return Path(__file__).parent.parent / "data" / "CUGA"
    
    # def format_trajectory(self, traj_data: dict) -> str:
    #     """Format a CUGA trajectory (short/full) into readable text."""
    #     lines = []
    #     trajectory = traj_data.get("trajectory", [])
    #     #print("=========")
    #     for i, step in enumerate(trajectory):
    #         step_num = step.get("step_number", "?")
    #         agent = step.get("agent_name", "Unknown")
    #         model_input = step.get("model_input", "")
    #         response = step.get("response", "")
    #
    #         lines.append(f"--- Step {step_num} [{agent}] ---")
    #         if model_input:
    #             #print(f"*{step.get('step_number', 0)} [{step.get('agent_name')}]: truncating {len(model_input)} -> 3000")
    #             input_text = model_input[:3000]
    #             if len(model_input) > 3000:
    #                 input_text += "\n... [truncated] ..."
    #             lines.append(f"INPUT:\n{input_text}")
    #         if response:
    #             resp_text = response[:3000]
    #             if len(response) > 3000:
    #                 resp_text += "\n... [truncated] ..."
    #             lines.append(f"RESPONSE:\n{resp_text}")
    #         lines.append("")
    #
    #     return "\n".join(lines)

    def extract_final_response(self, traj_data: dict) -> str:
        """Extract the final agent response from a CUGA trajectory."""
        trajectory = traj_data.get("steps", [])
        skip_agents = {"EvaluationResult"}
        
        # First pass: look for FinalAnswerAgent
        final_answer_response = ""
        for step in trajectory:
            if step.get("agent") == "FinalAnswerAgent":
                final_answer_response = step["timeline"][-1].get("text")
                if final_answer_response:
                    return final_answer_response
        
        # Second pass: last non-empty response from any non-skipped agent
        for step in reversed(trajectory):
            agent = step.get("agent", "")
            if agent in skip_agents:
                continue
            final_answer_response = step["timeline"][-1].get("text")
            if final_answer_response:
                return final_answer_response
        
        return final_answer_response

    def extract_user_request(self, traj_data: dict) -> str:
        """Return the task objective for a CUGA trajectory.

        Primary source is the top-level ``intent`` field.  Falls back to the
        first step's ``model_input`` if intent is empty.
        """
        intent = traj_data.get("intent", "").strip()
        if intent:
            return intent

        trajectory = traj_data.get("trajectory", [])
        if trajectory:
            first_input = trajectory[0].get("model_input", "").strip()
            if first_input:
                return first_input[:2000]

        return ""


# =============================================================================
# FinOps Dataset Implementation
# =============================================================================

class FinOpsDataset(DatasetBase):
    """FinOps dataset implementation."""

    name = "finops"

    def get_default_data_dir(self) -> Path:
        return Path(__file__).parent.parent / "data" / "finops"
    
    # def format_trajectory(self, traj_data: dict) -> str:
    #     """Format a FinOps trajectory into readable text."""
    #     lines = []
    #     trajectory = traj_data.get("trajectory", [])
    #     print()
    #     for i, step in enumerate(trajectory):
    #         task_name = step.get("task_name", "Unknown")
    #         step_type = step.get("type", "")
    #         elements = step.get("elements", [])
    #
    #         lines.append(f"--- {task_name} ({step_type}) ---")
    #         for elem in elements:
    #             elem_type = elem.get("type", "")
    #             message = elem.get("message", "")
    #             if message:
    #                 print(f"{i}: truncating message: {len(message)} -> 3000")
    #                 msg_text = message[:3000]
    #                 if len(message) > 3000:
    #                     msg_text += "\n... [truncated] ..."
    #                 lines.append(f"[{elem_type.upper()}]: {msg_text}")
    #         lines.append("")
    #
    #     return "\n".join(lines)
    
    def extract_user_request(self, traj_data: dict) -> str:
        """Extract the task objective from a FinOps trajectory."""
        import json as json_module
        
        metadata = traj_data.get("metadata", {})
        spans = metadata.get("spans", [])
        
        for span in spans:
            attrs = span.get("attributes", {})
            crew_tasks_raw = attrs.get("crew_tasks", "")
            if crew_tasks_raw:
                try:
                    crew_tasks = json_module.loads(crew_tasks_raw)
                    descriptions = []
                    for t in crew_tasks:
                        desc = t.get("description", "")
                        expected = t.get("expected_output", "")
                        if desc:
                            entry = desc.strip()
                            if expected:
                                entry += f" (Expected output: {expected.strip()})"
                            descriptions.append(entry)
                    if descriptions:
                        return " | ".join(descriptions)
                except (json_module.JSONDecodeError, TypeError):
                    pass
        
        # Fallback: first human input
        trajectory = traj_data.get("trajectory", [])
        for step in trajectory:
            elements = step.get("elements", [])
            for elem in elements:
                if elem.get("type", "").lower() in ("human_input", "input", "user", "prompt"):
                    msg = elem.get("message", "").strip()
                    if msg:
                        return msg
        
        return ""
    
    def extract_final_response(self, traj_data: dict) -> str:
        """Extract the final agent response from a FinOps trajectory."""
        trajectory = traj_data.get("trajectory", [])
        
        for step in reversed(trajectory):
            elements = step.get("elements", [])
            for elem in reversed(elements):
                msg = elem.get("message", "").strip()
                if msg:
                    return msg
        
        return ""



# =============================================================================
# WXO Dataset Implementation
# =============================================================================

class WXODataset(DatasetBase):
    """WXO dataset implementation."""

    name = "wx0"

    def get_default_data_dir(self) -> Path:
        return Path(__file__).parent.parent / "data" / "wxo"
    
    # def format_trajectory(self, traj_data: dict) -> str:
    #     """Format a WXO trajectory into readable text."""
    #     lines = []
    #     trajectory = traj_data.get("trajectory", [])
    #
    #     for i, step in enumerate(trajectory):
    #         role = step.get("role", "unknown")
    #         content = step.get("content", "")
    #         step_type = step.get("type", "")
    #         event = step.get("event", "")
    #
    #         type_info = f" type={step_type}" if step_type else ""
    #         event_info = f" event={event}" if event else ""
    #         lines.append(f"--- Step {i+1} [{role}{type_info}{event_info}] ---")
    #         if content:
    #             content_text = content[:3000]
    #             if len(content) > 3000:
    #                 content_text += "\n... [truncated] ..."
    #             lines.append(content_text)
    #         lines.append("")
    #
    #     return "\n".join(lines)

    def extract_user_request(self, traj_data: dict) -> str:
        """Extract the task objective from a WXO trajectory (first user message)."""
        trajectory = traj_data.get("trajectory", [])
        for step in trajectory:
            if step.get("role", "").lower() == "user":
                content = step.get("content", "").strip()
                if content:
                    return content
        return ""
    
    def extract_final_response(self, traj_data: dict) -> str:
        """Extract the final agent response from a WXO trajectory."""
        trajectory = traj_data.get("trajectory", [])
        for step in reversed(trajectory):
            if step.get("role", "").lower() == "assistant":
                content = step.get("content", "").strip()
                if content:
                    return content
        return ""


# =============================================================================
# Dataset Registry
# =============================================================================

class HALGenAgentDataset(DatasetBase):
    def extract_final_response(self, traj_data: dict) -> str:
        steps = traj_data.get("steps", [])
        if steps:
            final_response = steps[-1].get("output", {}).get("content", "")
            if final_response:
                return final_response
        return None

class TrailDataset(DatasetBase):
    def extract_final_response(self, traj_data: dict) -> str:
        return traj_data.get("final_answer")






DATASET_REGISTRY: Dict[str, Type[DatasetBase]] = {
    "CUGA": CUGADataset,
    "FinOps": FinOpsDataset,
    "WXO": WXODataset,
    "HAL": HALGenAgentDataset,
    "TRAIL": TrailDataset,
}


def get_dataset_obj(dataset_name: str, data_dir: Path = None) -> DatasetBase:
    """
    Get a dataset instance by name.
    
    Args:
        dataset_name: Name of the dataset (e.g., "CUGA", "FinOps", "WXO")
        data_dir: Optional override for data directory
        
    Returns:
        Dataset instance
        
    Raises:
        ValueError: If dataset name is not registered
    """
    if dataset_name not in DATASET_REGISTRY:
        available = ", ".join(DATASET_REGISTRY.keys())
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Available datasets: {available}"
        )
    
    dataset_class = DATASET_REGISTRY[dataset_name]
    return dataset_class(data_dir=data_dir)


def get_available_datasets() -> list:
    """Get list of available dataset names."""
    return list(DATASET_REGISTRY.keys())

