# from transformers import T5ForConditionalGeneration, T5Tokenizer

# # --- Load the Generative Model and Tokenizer (do this once at the start) ---
# print("\nLoading generative model (Flan-T5)...")
# model_name = "google/flan-t5-xl"
# tokenizer = T5Tokenizer.from_pretrained(model_name)
# model_gen = T5ForConditionalGeneration.from_pretrained(model_name)
# print("✅ Generative model loaded.")


# def generate_answer(query, context_chunks):
#     """
#     Takes the user's query and the retrieved context chunks, and generates a final answer.
#     """
#     # If context_chunks are dicts, extract their 'text' field
#     if isinstance(context_chunks[0], dict):
#         context_texts = [c["text"] for c in context_chunks if "text" in c]
#     else:
#         context_texts = context_chunks  # already a list of strings

#     # Combine the context chunks into a single string
#     context = " ".join(context_texts)

#     # Create the prompt using a template
#     prompt = f"""
#     Context: {context}
#     Question: {query}
#     Answer the question based only on the provided context.
#     Answer:
#     """

#     # Tokenize the prompt and generate the answer
#     inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
#     outputs = model_gen.generate(**inputs, max_length=150, num_beams=4, early_stopping=True)
#     answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

#     return answer


import os

# --- Load Groq Client (make sure you set GROQ_API_KEY in env) ---
from groq import Groq
from config import GROQ_API_KEY, require_env

MODEL_NAME = "llama-3.1-8b-instant"


def get_groq_client(require_key: bool = False):
    """Return a Groq client, constructing it lazily.

    If require_key is True, raise if GROQ_API_KEY is not set.
    """
    key = GROQ_API_KEY
    if require_key:
        require_env("GROQ_API_KEY", key)
    if not key:
        return None
    return Groq(api_key=key)


LAST_GENERATION_META = None


def _format_chat_history(chat_history) -> str:
    if not chat_history:
        return ""
    parts = ["Conversation history (most recent last):"]
    # chat_history is expected to be iterable of dicts with keys 'user' and 'assistant' and optional 'model'
    for turn in chat_history[-10:]:
        user = turn.get('user') if isinstance(turn, dict) else None
        assistant = turn.get('assistant') if isinstance(turn, dict) else None
        model = turn.get('model') if isinstance(turn, dict) else None
        if user:
            parts.append(f"User: {user}")
        if assistant:
            if model:
                parts.append(f"Assistant ({model}): {assistant}")
            else:
                parts.append(f"Assistant: {assistant}")
    return "\n".join(parts)


def generate_answer(query, context_chunks, chat_history=None):
    """
    Takes the user's query and retrieved context chunks, 
    then generates a final answer using Groq Llama-3.1-8b-instant.
    """
    # Backwards-compatible signature: allow optional segment by checking for
    # a third parameter if callers pass it positionally. Some callsites may
    # continue to pass only (query, chunks) so we keep this flexible.
    segment = 'factual'
    # If caller passed a tuple/list as context_chunks and included segment
    # (unlikely), ignore. We allow callers to pass a kwarg 'segment'.
    if isinstance(context_chunks, dict) and 'segment' in context_chunks:
        # defensive: if someone accidentally passed a dict with segment
        segment = context_chunks.get('segment', 'factual')
    # If function was called with 3 args (some callsites may), try to read it
    try:
        # inspect the calling frame arguments if available (best-effort)
        import inspect
        frame = inspect.currentframe().f_back
        args, _, _, values = inspect.getargvalues(frame)
        # if the caller provided a third positional arg, use it
        if len(args) >= 3 and args[2] in values:
            maybe_seg = values.get(args[2])
            if isinstance(maybe_seg, str):
                segment = maybe_seg
    except Exception:
        # ignore inspection errors
        pass
    # If context_chunks are dicts, extract their 'text' field
    if not context_chunks:
        return "⚠️ No context provided for generation."

    if isinstance(context_chunks[0], dict):
        context_texts = [c.get("text", "") for c in context_chunks if "text" in c]
    else:
        context_texts = context_chunks

    # Combine chunks into a single context string
    context = " ".join(context_texts)

    # Include chat history (formatted) so generators can keep context across turns
    history_text = _format_chat_history(chat_history)

    # Instruction-style prompt for the LLM
    prompt = f"""
You are a helpful tutor for school students. 
Answer based only on the provided context and the recent conversation history below.
If the context is insufficient, say clearly: "The context does not provide enough detail."

{history_text}

Context:
{context}

Question: {query}

Answer:
"""
    # Try to use the local GeneratorPool (multi-model) first for diversity.
    try:
        # package-qualified imports so tests can monkeypatch generation.generator_pool
        from generation.generator_pool import GeneratorPool
        from generation.ranker import rank_results
        from generation.fusion import fuse

        cfg = {
            "openrouter_url": os.getenv("OPENROUTER_URL"),
            "openrouter_key": os.getenv("OPENROUTER_KEY")
        }
        pool = GeneratorPool(config=cfg)
        # prepend chat history as a short context chunk so generators see it
        if history_text:
            context_with_history = [history_text] + context_texts
        else:
            context_with_history = context_texts

        # run multiple generators for this segment using the query+contexts
        multi_out = pool.run_all(query, context_with_history, segment, max_tokens=300)
        if multi_out:
            # multi_out: List[Tuple[name, text]]
            scored = rank_results(multi_out)
            strategy = os.getenv("GENERATION_FUSION", "best")
            final = fuse(scored, strategy=strategy)
            if final and final.strip():
                # If the fused/best output is just a local-echo (test/offline), prefer a reliable fallback
                # Determine whether the fused/best output is an echo. Prefer
                # to treat as echo only when the final text itself is an echo
                # or when all generator outputs are echoes.
                try:
                    final_is_echo = isinstance(final, str) and final.strip().startswith('[echo:')
                    all_echo = all(((isinstance(t, str) and t.strip().startswith('[echo:')) or ('local_echo' in (n or ''))) for n, t in multi_out)
                    is_echo = final_is_echo or all_echo
                except Exception:
                    is_echo = False

                # record meta for callers (which model(s) contributed)
                global LAST_GENERATION_META
                LAST_GENERATION_META = {"source": "pool", "models": [n for n, _ in multi_out]}

                if is_echo:
                    # If the fused/best output is an echo but there are non-echo
                    # candidates available, prefer the top non-echo candidate.
                    try:
                        if final_is_echo and scored:
                            for nm, txt, sc in scored:
                                if not (isinstance(txt, str) and txt.strip().startswith('[echo:')) and 'local_echo' not in (nm or ''):
                                    final = txt
                                    is_echo = False
                                    break
                    except Exception:
                        pass
                    # If we found a non-echo candidate, return it immediately
                    if not is_echo:
                        return final
                    # Try Groq fallback if available
                    client = get_groq_client()
                    if client is not None:
                        # fall back to calling Groq with the same prompt
                        try:
                            response = client.chat.completions.create(
                                model=MODEL_NAME,
                                messages=[
                                    {"role": "system", "content": "You are a knowledgeable tutor that explains concepts clearly."},
                                    {"role": "user", "content": prompt}
                                ],
                                temperature=0.7,
                                max_tokens=300,
                                top_p=0.9
                            )
                            ans = response.choices[0].message.content.strip()
                            LAST_GENERATION_META = {"source": "groq_fallback", "model": MODEL_NAME}
                            return ans
                        except Exception:
                            # fall through to returning concatenated contexts below
                            pass

                    # If no Groq client or it failed, return a helpful message with the top contexts
                    return "⚠️ No LLM backends available; here are the top retrieved contexts:\n\n" + "\n\n".join(context_texts[:3])

                return final
    except Exception as e:
        # if any multi-model machinery fails, fall back to single-model Groq flow
        print(f"⚠️ Multi-generator path failed: {e} — falling back to Groq client.")

    # Call Groq API (lazy client) as a reliable fallback
    client = get_groq_client()
    if client is None:
        return "⚠️ Generative API key not configured. Set GROQ_API_KEY in the environment."
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a knowledgeable tutor that explains concepts clearly."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=300,
        top_p=0.9
    )

    # Extract the model output
    answer = response.choices[0].message.content.strip()
    LAST_GENERATION_META = {"source": "groq", "model": MODEL_NAME}
    return answer


def grammar_correct(text: str) -> str:
    """Use LLM to correct grammar and preserve meaning. Returns corrected text."""
    prompt = f"""
You are a grammar assistant. Correct the user's text to fluent, well-punctuated English while preserving meaning.
Return only the corrected text.

Input:
{text}

Corrected:
"""
    try:
        client = get_groq_client()
        if client is None:
            return "⚠️ Generative API key not configured. Set GROQ_API_KEY in the environment."
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a helpful grammar assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=200
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"⚠️ Grammar generation error: {e}")
        return text


def summarize_context(context_chunks) -> str:
    """Summarize a list of context chunks into a concise student-friendly summary."""
    if not context_chunks:
        return "⚠️ No context provided for summarization."

    # Build context string
    if isinstance(context_chunks[0], dict):
        context_texts = [c.get("text", "") for c in context_chunks]
    else:
        context_texts = context_chunks

    context = "\n\n".join(context_texts)
    prompt = f"""
You are a helpful tutor. Summarize the following context into a concise, clear summary suitable for a middle/high school student. Keep it short (3-6 sentences).

Context:
{context}

Summary:
"""
    try:
        client = get_groq_client()
        if client is None:
            return "⚠️ Generative API key not configured. Set GROQ_API_KEY in the environment."
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a knowledgeable tutor who summarizes clearly."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=250
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"⚠️ Summarization error: {e}")
        return "⚠️ Summarization failed."


def faq_answer(query: str, retrieved_chunks) -> str:
    """Simple FAQ-style answering: use retrieved chunks and LLM to create concise answer.

    If no retrieved_chunks, returns a not-found message.
    """
    if not retrieved_chunks:
        return "⚠️ No FAQ entries found."

    # Use only top chunk for concise FAQ response
    top = retrieved_chunks[0]
    if isinstance(top, dict):
        top_text = top.get("text", "")
    else:
        top_text = top

    prompt = f"""
You are a helpful FAQ assistant. Given the following knowledge snippet, answer the user's question concisely.

Knowledge snippet:
{top_text}

Question: {query}

Answer concisely:
"""
    try:
        client = get_groq_client()
        if client is None:
            return "⚠️ Generative API key not configured. Set GROQ_API_KEY in the environment."
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a concise FAQ bot."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=200
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"⚠️ FAQ generation error: {e}")
        return "⚠️ FAQ lookup failed."


def image_understanding(question: str, retrieved_chunks) -> str:
    """Answer questions about diagrams/images by using nearby text chunks as context.

    This is a lightweight implementation: it uses the retrieved text chunks and
    asks the LLM to answer image-related questions.
    """
    if not retrieved_chunks:
        return "⚠️ No context found for image/diagram." 

    # Combine a few chunks for context
    if isinstance(retrieved_chunks[0], dict):
        context_texts = [c.get("text", "") for c in retrieved_chunks[:3]]
    else:
        context_texts = retrieved_chunks[:3]

    context = "\n\n".join(context_texts)
    prompt = f"""
You are an assistant that answers questions about diagrams and labelled figures using only the provided textual context. If the context does not mention the figure, say so.

Context:
{context}

Question: {question}

Answer:
"""
    try:
        client = get_groq_client()
        if client is None:
            return "⚠️ Generative API key not configured. Set GROQ_API_KEY in the environment."
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You answer diagram questions based on text context."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=300
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"⚠️ Image understanding error: {e}")
        return "⚠️ Image/diagram processing failed."
