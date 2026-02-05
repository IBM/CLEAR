from clear_eval.pipeline.use_cases.EvalUseCase import MathUseCase, RAGUseCase, GeneralEvalUseCase

def get_task_data_obj(use_case_name):
    if use_case_name == "math":
        return MathUseCase()
    elif use_case_name == "rag":
        return RAGUseCase()
    elif use_case_name == "general":
        return GeneralEvalUseCase()
    elif use_case_name == "tool_call":
        from clear_eval.pipeline.use_cases.ToolCallUseCase import ToolCallEvalUseCase
        return ToolCallEvalUseCase()
    elif use_case_name == "external" or use_case_name == "external_judge":
        from clear_eval.pipeline.use_cases.ExternalJudgeUseCase import ExternalJudgeUseCase
        return ExternalJudgeUseCase()
    raise ValueError(f"Unsupported use case: {use_case_name}")

