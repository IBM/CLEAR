import json


class LLMBackend:
    def chat(self, messages):
        raise NotImplementedError

    def embed(self, text):
        raise NotImplementedError


from urllib.parse import urlparse

# http_backend.py
import requests
from urllib.parse import urljoin

class HTTPBackend:
    def __init__(self, base_url: str, model_name: str, api_key: str | None = None):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.api_key = api_key

    def _headers(self):
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def chat(self, messages, stream: bool = False):
        url = urljoin(self.base_url + "/", "chat/completions")
        resp = requests.post(
            url,
            json={"model": self.model_name, "messages": messages, "stream": stream},
            headers=self._headers(),
            stream=stream,
            timeout=30
        )
        resp.raise_for_status()

        if not stream:
            j = resp.json()
            return j["choices"][0]["message"]["content"]

        def gen():
            for line in resp.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data: "):
                    continue
                payload = line[len("data: "):]
                if payload == "[DONE]":
                    break
                chunk = json.loads(payload)
                yield chunk["choices"][0]["delta"].get("content", "")
        return gen()

    def embed(self, texts):
        url = urljoin(self.base_url + "/", "embeddings")
        resp = requests.post(
            url,
            json={"model": self.model_name, "input": texts},
            headers=self._headers(),
            timeout=30
        )
        resp.raise_for_status()
        j = resp.json()
        return [row["embedding"] for row in j["data"]]

class WatsonXBackend(LLMBackend):
    def __init__(self, url, model):
        from ibm_watsonx_ai.foundation_models import Model
        self.model = Model(model_id=model)

    def chat(self, messages):
        prompt = messages[-1]["content"]
        return self.model.generate(prompt=prompt)


class LocalModelBackend(LLMBackend):
    def __init__(self, model_name):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def chat(self, messages):
        prompt = messages[-1]["content"]
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_new_tokens=128)
        return self.tokenizer.decode(outputs[0])


def load_backend(model_config):
    url = model_config["url"]
    name = model_config["name"]

    scheme = urlparse(url).scheme

    if scheme in ("http", "https"):
        return HTTPBackend(url, name)

    if scheme == "local":
        return LocalModelBackend(name)

    if scheme == "watsonx":
        return WatsonXBackend(url, name)

    raise ValueError(f"Unsupported scheme: {scheme}")



#x = load_backend({"url": "https://ete-litellm.bx.cloud9.ibm.com", "name": "openai/Azure/gpt-4.1"})
#url ="watsonx://us-south"
#name= "openai/gpt-oss-120b"

url = "http://127.0.0.1:8000/v1"
name = "Qwen/Qwen2.5-1.5B-Instruct"

config_http = {
    "model": {
        "url": url,
        "name": name,
        "api_key": None
    }
}

backend = load_backend(config_http["model"])

# Non-streaming chat
answer = backend.chat([
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Say hello in one word."}
])
print("Non-stream:", answer)

# Streaming chat
print("Stream:", end=" ")
for token in backend.chat([{"role": "user", "content": "Stream a short reply"}], stream=True):
    print(token, end="", flush=True)
print()
