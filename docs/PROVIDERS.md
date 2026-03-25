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
- **And many more**: See [LiteLLM providers](https://docs.litellm.ai/docs/providers)

**Configuration (new style):**
```yaml
provider: anthropic
inference_backend: litellm  # default, can be omitted
eval_model_name: claude-3-5-sonnet-20241022
```

**Configuration (old style - backward compatible):**
```yaml
provider: anthropic
use_litellm: true  # Maps to inference_backend: litellm
eval_model_name: claude-3-5-sonnet-20241022
```

**Example:**
```bash
export ANTHROPIC_API_KEY=your_key
run-clear-eval-analysis --config_path config.yaml
```

**Benefits:**
- **Async execution**: Better concurrency with async I/O
- **100+ providers**: Unified interface across providers
- **Flexible**: Easy to switch between providers

> ⚠️ **Credentials**: Set required environment variables for your provider according to [LiteLLM's documentation](https://docs.litellm.ai/docs/providers).

---

## 3. Direct HTTP Endpoint Backend

Use for custom APIs or when you need direct HTTP control. Works with any OpenAI-compatible endpoint (vLLM, Ollama, etc.):

**Configuration:**
```yaml
provider: watsonx  # or openai, anthropic
inference_backend: endpoint
endpoint_url: https://us-south.ml.cloud.ibm.com/ml/v1
eval_model_name: ibm/granite-3-8b-instruct
```

**Supported endpoint types:**
- `openai`: OpenAI API and compatible (vLLM, Ollama, local servers)
- `watsonx`: IBM WatsonX

**Example with local vLLM:**
```yaml
provider: openai
inference_backend: endpoint
endpoint_url: http://localhost:8000/v1
eval_model_name: your-model-name
```

---

## Quick Reference

| Provider | Backend | Credentials |
|----------|---------|-------------|
| OpenAI | LangChain, LiteLLM, Endpoint | `OPENAI_API_KEY` |
| WatsonX | LangChain, LiteLLM, Endpoint | `WATSONX_APIKEY`, `WATSONX_URL`, `WATSONX_PROJECT_ID` |
| Anthropic | LiteLLM | `ANTHROPIC_API_KEY` |
| AWS Bedrock | LiteLLM | AWS credentials |
| Google Vertex AI | LiteLLM | GCP credentials |
| [100+ more](https://docs.litellm.ai/docs/providers) | LiteLLM | Provider-specific |
