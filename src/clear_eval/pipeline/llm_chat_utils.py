import os
from langchain_openai import AzureChatOpenAI, ChatOpenAI


def get_chat_llm(provider, model_id, parameters=None, eval_mode = True):
    if parameters is None:
        parameters = {}
    if provider == "watsonx":
        from langchain_ibm import ChatWatsonx
        parameters = parameters or {
            "min_new_tokens": 1,
            "stop_sequences": [".", "<|eom_id|>"],
            "enable-auto-tool-choice": False,
            "tool-call-parser": False
        }
        if eval_mode:
            parameters["decoding_method"]= "greedy"
        space_id = os.getenv("WATSONX_SPACE_ID")
        if space_id:
            return ChatWatsonx(
                model_id=model_id,
                url=os.getenv("WATSONX_URL"),
                apikey=os.getenv("WATSONX_APIKEY"),
                space_id=space_id,
                params=parameters,
            )
        project_id = os.getenv("WATSONX_PROJECT_ID")
        if project_id:
            return ChatWatsonx(
                model_id=model_id,
                url=os.getenv("WATSONX_URL"),
                apikey=os.getenv("WATSONX_APIKEY"),
                project_id=project_id,
                params=parameters,
            )
        raise KeyError("Either WATSONX_SPACE_ID or WATSONX_PROJECT_ID must be specified for watsonx inference.")
    if provider == "rits":
        model_base = model_name_to_rits_base.get(model_id)
        if eval_mode:
            parameters["temperature"] = 0
        if not model_base:
            model_base = model_id.split("/")[-1].replace(".", "-").lower().replace("-vision", "")
        return ChatOpenAI(
            model=model_id,
            api_key='/',
            base_url=f'https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/{model_base}/v1',
            default_headers={'RITS_API_KEY': os.getenv("RITS_API_KEY")},
            max_retries=2,
            **parameters
        )
    if provider == "azure":
        azure_openapi_host = os.getenv("AZURE_OPENAI_HOST")
        api_version = os.getenv("OPENAI_API_VERSION")
        model_id = model_id or "gpt-4o-2024-08-06"
        model_base = model_id.split("/")[-1]
        azure_endpoint = f'{azure_openapi_host}/openai/deployments/{model_base}/chat/completions?api-version={api_version}'
        if eval_mode:
            parameters["temperature"] = 0
        return AzureChatOpenAI(api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                               azure_endpoint=azure_endpoint,
                               api_version=api_version,
                               max_retries=2,
                               **parameters
                               )
    if provider == "openai":
        if eval_mode:
            parameters["temperature"] = 0
        model_id = model_id
        return ChatOpenAI(
            model=model_id,
            max_retries=2,
            **parameters

        )
    raise ValueError(f"Unknown provider {provider}, supported providers: watsonx, rits, azure or openai")

model_name_to_rits_base = {
    "microsoft/phi-4": "microsoft-phi-4",
    "microsoft/Phi-4-reasoning": "phi-4-reasoning",
    "mistralai/mixtral-8x7B-instruct-v0.1": "mixtral-8x7b-instruct-v01",
    "mistralai/mixtral-8x22B-instruct-v0.1": "mixtral-8x22b-instruct-v01",
    "meta-llama/llama-4-maverick-17b-128e-instruct-fp8": "llama-4-mvk-17b-128e-fp8",
    "deepseek-ai/DeepSeek-V3": "deepseek-v3-h200",
    "meta-llama/Llama-3.1-8B-Instruct": "llama-3-1-8b-instruct",
    "meta-llama/Llama-4-Scout-17B-16E-Instruct": "llama-4-scout-17b-16e-instruct",
    "mistralai/Mistral-Small-3.1-24B-Instruct-2503": "mistral-small-3-1-24b-2503",
    "ibm-granite/granite-guardian-3.2-5b": "granite-guardian-3-2-5b-ris",
}