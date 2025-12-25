import os
import traceback
from flask import Flask, request, jsonify

# Prefer to reuse the main orchestration (which provides ChatMemory and
# initialization helpers). Fall back to importing run_pipeline directly
# if main helpers aren't available.
try:
    import main
    SERVER_STATE = main.initialize_server()
    HANDLE_QUERY = main.handle_query
    run_pipeline = None
except Exception:
    SERVER_STATE = None
    HANDLE_QUERY = None
    try:
        from pipeline.segment_pipeline import run_pipeline
    except Exception:
        run_pipeline = None

# Per-session memory store: mapping session_id -> ChatMemory instance
from uuid import uuid4
SESSION_MEMS = {}

app = Flask(__name__)


@app.route("/", methods=["GET"])
def index():
    return "TechItChatBot backend is running. POST /chat with JSON {query, chat_history} to get an answer."


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    query = data.get("query") or data.get("q") or ""
    chat_history = data.get("chat_history")

    if not query:
        return jsonify({"error": "No query provided."}), 400

    # If we have a running server state (from main.initialize_server), use it
    if SERVER_STATE is not None and HANDLE_QUERY is not None:
        try:
            # support per-session memory: client may provide 'session_id'
            session_id = data.get("session_id")
            if session_id and session_id in SESSION_MEMS:
                mem = SESSION_MEMS[session_id]
            else:
                # create a new session if none provided or unknown
                session_id = session_id or str(uuid4())
                try:
                    mem = main.ChatMemory(persist_path=None)
                except Exception:
                    # fallback: use the default server mem
                    mem = SERVER_STATE.get('mem')
                SESSION_MEMS[session_id] = mem

            # Build a per-session server dict that reuses heavy components
            server_for_session = {
                'mem': mem,
                'run_pipeline': SERVER_STATE.get('run_pipeline'),
                'kgq': SERVER_STATE.get('kgq'),
                'gen_mod': SERVER_STATE.get('gen_mod')
            }

            answer, meta = HANDLE_QUERY(server_for_session, query)
        except Exception as e:
            tb = traceback.format_exc()
            print(f"Backend /chat error (server): {e}\n{tb}")
            return jsonify({"error": str(e), "trace": tb}), 500
        return jsonify({"answer": answer, "meta": meta, "session_id": session_id})

    # Otherwise fall back to calling run_pipeline directly using any provided chat_history
    if run_pipeline is None:
        return jsonify({"error": "Pipeline import failed on backend. See server logs."}), 500

    try:
        # Prefer the newer API that accepts chat_history
        try:
            answer = run_pipeline(query, chat_history=chat_history)
        except TypeError:
            # older signature: run_pipeline(query)
            answer = run_pipeline(query)
    except Exception as e:
        tb = traceback.format_exc()
        print(f"Backend /chat error: {e}\n{tb}")
        return jsonify({"error": str(e), "trace": tb}), 500

    # Try to include generation meta if available
    meta = {}
    try:
        from generation.generate_answers import LAST_GENERATION_META
        meta = LAST_GENERATION_META if LAST_GENERATION_META is not None else {}
    except Exception:
        meta = {}

    return jsonify({"answer": answer, "meta": meta})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    debug = os.environ.get("FLASK_DEBUG", "1") in ("1", "true", "True")
    print(f"Starting Flask backend on 0.0.0.0:{port} (debug={debug})")
    app.run(host="0.0.0.0", port=port, debug=debug)
