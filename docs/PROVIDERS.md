# Supported Providers and Credentials

CLEAR supports three inference backends for maximum flexibility. This configuration applies to both [LLM Analysis](ANALYSIS_README.md) and [Agentic Analysis](../src/clear_eval/agentic/README.md).

---

## 1. LangChain Backend

Use for built-in providers with LangChain integration:

| Provider   | Required Environment Variables                                              |
|------------|-----------------------------------------------------------------------------|
| `openai`   | `OPENAI_API_KEY`,  [`OPENAI_API_BASE` if using proxy ]                      |
| `watsonx`  | `WATSONX_APIKEY`, `WATSONX_URL`, `WATSONX_SPACE_ID` or `WATSONX_PROJECT_ID` |

**Configuration:**
```yaml
provider: watsonx
inference_backend: langchain
eval_model_name: meta-llama/llama-3-3-70b-instruct
```

---

## 2. LiteLLM Backend (Default)

Use for 100+ LLM providers via [LiteLLM](https://docs.litellm.ai/docs/providers):

- **Anthropic**: `anthropic` (Claude)
- **AWS Bedrock**: `bedrock`
- **Google**: `vertex_ai`, `gemini`
- **Groq**: `groq`
- **Together AI**: `together_ai`
- **Local models**: `openai` (vLLM, llama.cpp), `ollama`, `huggingface` (TGI)
- **And many more**: See [LiteLLM providers](https://docs.litellm.ai/docs/providers)

**Configuration:**
```yaml
provider: anthropic
inference_backend: litellm  # default, can be omitted
eval_model_name: claude-3-5-sonnet-20241022
```

**Configuration (local endpoint):**
```yaml
provider: openai            # or ollama, huggingface, etc.
inference_backend: litellm
endpoint_url: "http://localhost:8000/v1"
eval_model_name: my-local-model
```

**Configuration (old style - backward compatible):**
```yaml
provider: anthropic
use_litellm: true  # Maps to inference_backend: litellm
eval_model_name: claude-3-5-sonnet-20241022
```

> ⚠️ **Credentials**: Set required environment variables for your provider according to [LiteLLM's documentation](https://docs.litellm.ai/docs/providers). For local endpoints with `endpoint_url`, no credentials are needed.

