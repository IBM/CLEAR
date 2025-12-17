import json
import os
from typing import Tuple, Any, List, Dict
import pandas as pd
from clear_eval.pipeline.use_cases.EvalUseCase import EvalUseCase
from clear_eval.pipeline.constants import EVALUATION_TEXT_COL, SCORE_COL

import logging

from pipeline.config_loader import load_config
from pipeline.llm_chat_utils import get_chat_llm

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

        examples = []
        for i, row in df.iterrows():
            examples.append({self.ID_COL:row[self.ID_COL],
                             self.CONTEXT_COL: json.loads(row[self.CONTEXT_COL]),
                             self.SPECS_COL: json.loads(row[self.SPECS_COL]),
                             self.RESPONSE_COL: json.loads(row[self.RESPONSE_COL])})

        # call spark with pipeline over examples, results store sorted results over the examples
        results = self.generate_spark_evaluation_results(examples,
                                                         provider = config.get("provider"),
                                                         model_name = config.get("eval_model_name"))

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


    @staticmethod
    def generate_evaluation_model_prompt(row, config):
        return None

    @staticmethod
    def get_default_generation_model_inputs(row, config):
        raise NotImplementedError("Tool Call generations must be provided")

    def get_eval_from_results(self, result: Any) -> Tuple[str, float]:
        # TODO: IMPLEMENT
        return "Correct tool call", 1

    def generate_spark_evaluation_results(self, examples: List[Dict[str, Any]], provider: str, model_name: str):
        """Generates spark evaluation results."""
        # examples - List[Dict[str, Any]], keys: context, api_spec, response (the tool call), values: the json objects
        # provider - rits/watsonx/openai

        # TODO
        # spark_llm = get_spark_llm(provider, model_name)
        # construct input examples from df (input fields spec_cols and context_col given)
        # construct spark pipeline
        return [None] * len(examples)


if __name__ == "__main__":
    DEFAULT_CONFIG_PATH = os.path.join("..", "setup", "default_config.yaml")
    provider = "rits"  # watsonx, openai
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