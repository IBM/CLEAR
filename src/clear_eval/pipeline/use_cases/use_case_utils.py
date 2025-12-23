from clear_eval.pipeline.use_cases.EvalUseCase import MathUseCase, RAGUseCase, GeneralEvalUseCase
from clear_eval.pipeline.use_cases.ToolCallUseCase import ToolCallEvalUseCase

task_to_use_case_class = {
    "math": MathUseCase,
    "rag": RAGUseCase,
    "general": GeneralEvalUseCase,
    "tool_call": ToolCallEvalUseCase
}

def get_use_case_class(use_case_name):
    try:
        return task_to_use_case_class[use_case_name]
    except KeyError:
        raise ValueError(f"Unsupported use case: {use_case_name}")


def get_task_data_obj(task):
    return get_use_case_class(task)()

def get_supported_use_case_classes():
    return list(task_to_use_case_class.keys())
