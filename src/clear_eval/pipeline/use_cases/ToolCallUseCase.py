import json
from importlib.resources import files
from typing import Tuple, Any, List, Dict
import numpy as np
import pandas as pd
from clear_eval.pipeline.use_cases.EvalUseCase import EvalUseCase
from clear_eval.pipeline.constants import EVALUATION_TEXT_COL, SCORE_COL
from altk.pre_tool.sparc import SPARCReflectionComponent
from altk.core.toolkit import AgentPhase, ComponentConfig
from altk.pre_tool.core import SPARCExecutionMode, SPARCReflectionRunInput, Track, SPARCReflectionResult
from altk.core.llm import get_llm, BaseLLMClient

import logging

from clear_eval.pipeline.config_loader import load_config
from clear_eval.pipeline.llm_chat_utils import get_chat_llm

logger = logging.getLogger(__name__)

class ToolCallEvalUseCase(EvalUseCase):

    SPECS_COL = "api_spec"
    CONTEXT_COL = "context"
    RESPONSE_COL = "response"
    ID_COL = "id"

    def eval_records(self, df, llm, config, score_col = SCORE_COL):
        """Evaluates predictions and adds scores."""
        logger.info(f"\n--- Evaluating Tool calls predictionsPredictions ---")
        df[EVALUATION_TEXT_COL] = ""
        df[score_col] = pd.NA  # Use Pandas NA for missing scores

        # convert CLEAR llm to ALTK llm
        altk_llm_client = self.clear_llm_client_to_altk_llm_client(llm, config.get("provider"), config.get("eval_model_name"))

        # call sparc with pipeline over examples, results store sorted results over the examples
        results = self.generate_sparc_evaluation_results(df=df, llm_client=altk_llm_client)

        # extract output score and evaluation text from each results (concatenate failing explanations? minimum/average score over metrics?)
        for i, result in enumerate(results):
            (eval_text, score) = self.get_eval_from_results(result)  # TODO extract eval text and score from results
            score = score if pd.isna(score) else float(score)

            df.at[df.index[i], EVALUATION_TEXT_COL] = eval_text
            df.at[df.index[i], score_col] = score if pd.isna(score) else float(score)

        logger.info("Finished evaluating predictions.")
        # Convert score column to nullable float type
        df[score_col] = df[score_col].astype('Float64')
        return df
    

    def clear_llm_client_to_altk_llm_client(self, llm, provider: str, model_name: str) -> BaseLLMClient:
        """Convert CLEAR's LLM object to ALTK's LLM Object."""

        if provider == "watsonx":
            MetricsClientCls = get_llm("watsonx.output_val")
            if llm.space_id:
                llm_client = MetricsClientCls(
                    model_id=llm.model_id,
                    api_key=llm.api_key._secret_value,
                    url=llm.url,
                    space_id=llm.space_id,
                )
            elif llm.project_id:
                llm_client = MetricsClientCls(
                    model_id=llm.model_id,
                    api_key=llm.api_key._secret_value,
                    url=llm.url._secret_value,
                    project_id=llm.project_id
            )
            else:
                raise KeyError("Either space_id or project_id must be specified for watsonx inference.")
            
        elif provider == "azure":
            MetricsClientCls = get_llm("azure_openai.async.output_val")
            llm_client = MetricsClientCls(
                model=model_name,
                api_key=llm.api_key._secret_value,
            )
        elif provider == "openai":
            MetricsClientCls = get_llm("openai.async.output_val")
            llm_client = MetricsClientCls(
                model=llm.model,
            )
        
            llm_client = MetricsClientCls(model_name=model_name, api_key="")
        else:
            raise ValueError(f"Unsupported provider: {provider}")
        
        return llm_client


    @staticmethod
    def generate_evaluation_model_prompt(row, config):
        return None

    @staticmethod
    def get_default_generation_model_inputs(row, config):
        raise NotImplementedError("Tool Call generations must be provided")

    def get_eval_from_results(self, result: SPARCReflectionResult) -> Tuple[str, float]:
        """
        Compute grade from LLMEvalKit pipeline result according to the specified logic.

        Args:
            result: PipelineResult from LLMEvalKit

        Returns:
            Grade as float in [0, 1]
        """
        logger.debug("=== DEBUG: Full result structure ===")
        logger.debug(result)
        if result.decision.name == "APPROVE":
            return "Tool call is valid.", 1.0
        else:
            explanation_text = "\n".join([issue.explanation for issue in result.issues])
            logger.debug("=== DEBUG: Explanation Text ===")
            logger.debug(explanation_text)
            return f"Tool call is invalid. Reasons:\n{explanation_text}", 0.0
    

    def generate_sparc_evaluation_results(self, df: pd.DataFrame, llm_client: BaseLLMClient) -> List[SPARCReflectionResult]:
        """Generates sparc evaluation results."""
        sparc_component = SPARCReflectionComponent(
            config=ComponentConfig(llm_client=llm_client),
            track=Track.SLOW_TRACK,  # Use slow track for performance
            execution_mode=SPARCExecutionMode.ASYNC,
        )
        reflection_results = []
        for _, example in df.iterrows():
            run_input = SPARCReflectionRunInput(
                    messages=json.loads(example[self.CONTEXT_COL]),
                    tool_specs=json.loads(example[self.SPECS_COL]),
                    tool_calls=[json.loads(example[self.RESPONSE_COL])],
                )
            reflection_result = sparc_component.process(run_input, phase=AgentPhase.RUNTIME).output.reflection_result
            reflection_results.append(reflection_result)
        
        return reflection_results


if __name__ == "__main__":
    DEFAULT_CONFIG_PATH = str(files("clear_eval.pipeline.setup").joinpath("default_config.yaml"))
    provider = "watsonx"
    model_name = "meta-llama/llama-3-3-70b-instruct"
    config = load_config(DEFAULT_CONFIG_PATH, user_config_path=None, provider=provider, eval_model_name=model_name)

    test_examples = [
        # Example 1: valid get_weather
        {
            "call": {
                "id": "1",
                "type": "function",
                "function": {
                    "name": "get_weather_celsius",
                    "arguments": json.dumps({"location": "Berlin"}),
                },
            },
            "context": [{"role": "user", "content": "What's the weather in Berlin today?"}],
        },
        # Example 2: missing required 'time'
        {
            "call": {
                "id": "2",
                "type": "function",
                "function": {
                    "name": "create_event",
                    "arguments": json.dumps({"title": "Team Sync"}),
                },
            },
            "context": [
                {
                    "role": "user",
                    "content": "Schedule a meeting tomorrow at 10 AM called 'Team Sync'.",
                }
            ],
        },
        # Example 3: valid translate_text
        {
            "call": {
                "id": "3",
                "type": "function",
                "function": {
                    "name": "translate_text",
                    "arguments": json.dumps(
                        {"text": "Hello, world!", "target_language": "fr"}
                    ),
                },
            },
            "context": [
                {"role": "user", "content": "Please translate 'Hello, world!' into French."}
            ],
        },
        # Example 4: invalid units transformation
        {
            "call": {
                "id": "4",
                "type": "function",
                "function": {
                    "name": "comment_list",
                    "arguments": json.dumps(
                        {
                            "aweme_id": 456789123,
                            "count": 15,
                            "from": 1714521700,
                            "to": 1715299300,
                        }
                    ),
                },
            },
            "context": [
                {
                    "role": "user",
                    "content": "Fetch the first 15 comments for the IBM video with ID 456789123 from May 1st, 2024 to May 10th, 2024.",
                }
            ],
        },
        # Example 5: Invalid comment_list call with extra parameters, missing required 'aweme_id', and incorrect 'comments_type' with invalid value (Enum violation)
        {
            "call": {
                "id": "5",
                "type": "function",
                "function": {
                    "name": "comment_list",
                    "arguments": json.dumps(
                        {
                            "count": "15",
                            "from": 1714521700,
                            "to": 1715299300,
                            "sdfs": "sdf",
                            "comments_type": "ggg",
                        }
                    ),
                },
            },
            "context": [
                {
                    "role": "user",
                    "content": "Fetch the first 15 comments for the IBM video with ID 456789123 from May 1st, 2024 to May 10th, 2024.",
                }
            ],
        },
        # Example 6: Type conversion test - string to integer conversion
        {
            "call": {
                "id": "6",
                "type": "function",
                "function": {
                    "name": "comment_list",
                    "arguments": json.dumps(
                        {
                            "aweme_id": "456789123",  # String that should convert to int
                            "count": "10",  # String that should convert to int
                            "cursor": "0",  # String that should convert to int
                        }
                    ),
                },
            },
            "context": [
                {
                    "role": "user",
                    "content": "Get 10 comments for video 456789123 starting from the beginning.",
                }
            ],
        },
        # Example 7: Mixed type conversions with create_event
        {
            "call": {
                "id": "7",
                "type": "function",
                "function": {
                    "name": "create_event",
                    "arguments": json.dumps(
                        {
                            "title": 123,  # Number that should convert to string
                            "time": "2024-12-25T10:00:00Z",  # Valid string (should remain)
                            "attendees": "john@example.com",  # Single string that should convert to array
                        }
                    ),
                },
            },
            "context": [
                {
                    "role": "user",
                    "content": "Create an event called 123 on Christmas morning with john@example.com attending.",
                }
            ],
        },
    ]
    apis_specs: List[Dict[str, Any]] = [
        {
            "type": "function",
            "function": {
                "name": "get_weather_celsius",
                "description": "Retrieve current weather information for a given location in Celsius",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "Name of the city or geographic location",
                        }
                    },
                    "required": ["location"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "create_event",
                "description": "Schedule a calendar event with title, time, and optional attendees",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "title": {
                            "type": "string",
                            "description": "The event title or summary",
                        },
                        "time": {
                            "type": "string",
                            "format": "date-time",
                            "description": "ISO-8601 formatted start time",
                        },
                        "attendees": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of attendee email addresses",
                        },
                    },
                    "required": ["title", "time"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "translate_text",
                "description": "Translate a piece of text into a target language",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "The input text to translate",
                        },
                        "target_language": {
                            "type": "string",
                            "description": "The language code to translate into (e.g., 'fr', 'es')",
                        },
                    },
                    "required": ["text", "target_language"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "comment_list",
                "description": "Fetches a list of comments for a specified IBM video using the given API.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "aweme_id": {
                            "type": "integer",
                            "description": "The ID of the IBM video.",
                        },
                        "cursor": {
                            "type": "integer",
                            "description": "The cursor for pagination to get the next page of comments.",
                        },
                        "count": {
                            "type": "integer",
                            "description": "The number of comments to fetch. Maximum is 30.",
                        },
                        "comments_type": {
                            "type": "string",
                            "description": "Comments types.",
                            "enum": ["all", "good", "bad"],
                        },
                        "from": {
                            "type": "integer",
                            "description": "The start of the requested timeframe as a Unix timestamp.",
                        },
                        "to": {
                            "type": "integer",
                            "description": "The end of the requested timeframe as a Unix timestamp.",
                        },
                    },
                    "required": ["aweme_id"],
                },
            },
        },
    ]

    df = pd.DataFrame.from_records(test_examples).rename(columns={"call": ToolCallEvalUseCase.RESPONSE_COL,
                                                             "context": ToolCallEvalUseCase.CONTEXT_COL})
    df[ToolCallEvalUseCase.SPECS_COL] = json.dumps(apis_specs)
    df[ToolCallEvalUseCase.RESPONSE_COL] = df[ToolCallEvalUseCase.RESPONSE_COL].apply(json.dumps)
    df[ToolCallEvalUseCase.CONTEXT_COL] = df[ToolCallEvalUseCase.CONTEXT_COL].apply(json.dumps)
    df[ToolCallEvalUseCase.ID_COL] = df.index.astype(str)
    llm = get_chat_llm(config["provider"], config["eval_model_name"], eval_mode=True)

    tool_call_use_case = ToolCallEvalUseCase()
    evaluated_df = tool_call_use_case.eval_records(df, llm, config)