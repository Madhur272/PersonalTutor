"""Generator pool: manage multiple free LLM backends and call them with prompt templates.

Design:
- Each generator is a small adapter exposing generate(prompt, max_tokens, **kwargs) -> str
- We provide two free/test adapters: openrouter_adapter (HTTP, mockable) and local_echo (returns prompt sniff)
- GeneratorPool exposes get_generators_for_segment(segment) to pick which generators to call,
  and run_all(query, context, segment) which returns a list of (model_name, text)

This module is intentionally small and easily mockable for unit tests.
"""
from typing import List, Dict, Any, Tuple
import random
import os
from config import OPENROUTER_KEY, OPENROUTER_MODELS, OPENROUTER_MODEL_PARAMS


class BaseGenerator:
    def generate(self, prompt: str, max_tokens: int = 256, **kwargs) -> str:
        raise NotImplementedError()


class LocalEchoGenerator(BaseGenerator):
    """A trivial generator used for unit tests and offline fallback."""

    def __init__(self, name: str = "local_echo"):
        self.name = name

    def generate(self, prompt: str, max_tokens: int = 256, **kwargs) -> str:
        # Return a deterministic echo with a short prefix to simulate generation
        snippet = prompt[:200].replace('\n', ' ')
        return f"[echo:{self.name}] {snippet}"


class OpenRouterGenerator(BaseGenerator):
    """Minimal OpenRouter-compatible adapter. In tests this will be monkeypatched.

    Real usage should set OPENROUTER_API to point to a free-compatible router.
    This adapter accepts a model_name which is passed to the router so multiple
    distinct model backends can be used in parallel.
    """

    def __init__(self, model_name: str, name: str = None, api_url: str = None, api_key: str = None):
        self.model_name = model_name
        self.name = name or f"openrouter:{model_name}"
        self.api_url = api_url or os.getenv("OPENROUTER_URL") or "https://openrouter.ai/v1/chat/completions"
        self.api_key = api_key or OPENROUTER_KEY

    # def generate(self, prompt: str, max_tokens: int = 256, **kwargs) -> str:
    #     # Lightweight HTTP call; production code should handle retries/backoff.
    #     import requests
    #     payload = {
    #         "model": self.model_name,
    #         "input": prompt,
    #         "max_tokens": max_tokens,
    #     }
    #     # Accept optional generation params (temperature, top_p, stop, etc.) via kwargs
    #     # and merge them into payload so different models can be sampled differently.
    #     for k, v in kwargs.items():
    #         # only include simple serializable params
    #         if v is None:
    #             continue
    #         payload[k] = v
    #     headers = {"Content-Type": "application/json"}
    #     if self.api_key:
    #         headers["Authorization"] = f"Bearer {self.api_key}"
    #     resp = requests.post(self.api_url, json=payload, headers=headers, timeout=20)
    #     resp.raise_for_status()
    #     data = resp.json()
    #     # attempt to extract a sensible text field
    #     text = None
    #     if isinstance(data, dict):
    #         # openrouter-like responses might nest choices -> message -> content
    #         if "choices" in data and len(data["choices"]) > 0:
    #             ch = data["choices"][0]
    #             if isinstance(ch, dict) and "message" in ch:
    #                 msg = ch["message"]
    #                 if isinstance(msg, dict):
    #                     # content may be a string or dict
    #                     if isinstance(msg.get("content"), dict):
    #                         text = msg["content"].get("text") or str(msg["content"])
    #                     else:
    #                         text = msg.get("content") or msg.get("text")
    #             if not text:
    #                 text = ch.get("text")
    #     if not text:
    #         # fallback: stringify payload
    #         text = str(data)
    #     return text

    def generate(self, prompt: str, max_tokens: int = 256, **kwargs) -> str:
        import requests
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
        }
        payload.update({k: v for k, v in kwargs.items() if v is not None})

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"    

        resp = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            json=payload,
            headers=headers,
            timeout=20
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]


class GeneratorPool:
    def __init__(self, config: Dict[str, Any] = None):
        # config can specify API URLs/keys per generator
        self.config = config or {}
        # seed randomness optionally for reproducibility
        seed = self.config.get("random_seed")
        if seed is not None:
            random.seed(seed)
        # create a local echo fallback
        self.generators_base = {"local_echo": LocalEchoGenerator()}
        # prepare available openrouter models
        self.available_models = self.config.get("openrouter_models") or OPENROUTER_MODELS
        self.openrouter_key = self.config.get("openrouter_key") or OPENROUTER_KEY
        # model-specific parameter defaults (temperature, top_p, etc.)
        self.model_params = self.config.get("openrouter_model_params") or OPENROUTER_MODEL_PARAMS or {}

    def get_generators_for_segment(self, segment: str) -> List[BaseGenerator]:
        """Return a prioritized list of generators for a given segment.

        Simple default mapping: factual -> [openrouter, local_echo], explain -> [openrouter, local_echo], grammar -> [local_echo]
        """
        seg = segment.lower() if segment else "factual"
        # For each segment choose a random 2-3 models from available_models
        num = 3 if seg in {"factual", "explain", "summary"} else 2
        num = min(num, max(1, len(self.available_models)))
        chosen = random.sample(self.available_models, k=num)

        gens: List[BaseGenerator] = []
        # create OpenRouterGenerator instances for chosen models
        for m in chosen:
            # create generator and keep model-specific params accessible
            g = OpenRouterGenerator(model_name=m, api_key=self.openrouter_key)
            # attach params to generator for run time use
            setattr(g, "model_params", self.model_params.get(m, {}))
            gens.append(g)

        # always append a local echo fallback at the end
        gens.append(self.generators_base["local_echo"])
        return gens

    # Prompt templates per segment. Templates receive {context} and {query}.
    PROMPT_TEMPLATES = {
        "factual": (
            "You are a precise factual tutor. Use ONLY the provided context to answer the question. "
            "If the context is insufficient, say clearly: 'The context does not provide enough detail.'\n\n"
            "Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        ),
        "explain": (
            "You are an explanatory tutor. Given the context, explain the underlying concepts in a clear, step-by-step way suitable for a student.\n\n"
            "Context:\n{context}\n\nQuestion: {query}\n\nExplanation:"
        ),
        "summary": (
            "You are a concise summarizer. Produce a 3-6 sentence summary of the provided context focused on the question.\n\n"
            "Context:\n{context}\n\nSummarize with respect to: {query}\n\nSummary:"
        ),
        "grammar": (
            "You are a grammar assistant. Correct the user's text to fluent, well-punctuated English while preserving meaning.\n\n"
            "Input:\n{query}\n\nCorrected:\n"
        ),
        "faq": (
            "You are an FAQ assistant. Use the provided knowledge snippets to answer the question concisely. If not found, say so.\n\n"
            "Context:\n{context}\n\nQuestion: {query}\n\nAnswer concisely:"
        ),
        # default template
        "default": (
            "You are a helpful tutor. Use the provided context to answer the question.\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        )
    }

    def build_prompt_for_segment(self, segment: str, query: str, contexts: List[str]) -> str:
        seg = (segment or "").lower()
        tmpl = self.PROMPT_TEMPLATES.get(seg) or self.PROMPT_TEMPLATES.get("default")
        # combine contexts (keep them short)
        context_join = "\n\n".join([c.strip() for c in contexts if c])
        return tmpl.format(context=context_join, query=query)

    def run_all(self, query: str, contexts: List[str], segment: str, max_tokens: int = 256) -> List[Tuple[str, str]]:
        """Run all selected generators for a given segment using segment-specific prompts.

        Arguments:
            query: user query string
            contexts: list of context chunk texts
            segment: segment label to choose templates and models
        Returns: list of (generator_name, generated_text)
        """
        gens = self.get_generators_for_segment(segment)
        prompt = self.build_prompt_for_segment(segment, query, contexts)
        results: List[Tuple[str, str]] = []
        for g in gens:
            try:
                # per-model overrides passed into generate
                params = getattr(g, "model_params", {}) or {}
                txt = g.generate(prompt, max_tokens=max_tokens, **params)
            except Exception as e:
                txt = f"[error:{getattr(g,'name',str(g))}] {e}"
            results.append((getattr(g, 'name', g.__class__.__name__), txt))
        return results
