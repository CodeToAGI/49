"""EP49 - A chatbot with persistent memory: LangChain + LangGraph + any chat LLM + Gradio.
    python ep49_chatbot.py            # web UI
    python ep49_chatbot.py --cli      # terminal chat  (type /forget to wipe memory, quit to stop)
pip install langchain-core langchain-huggingface transformers torch gradio
pip install langgraph langgraph-checkpoint-sqlite
Optional API model:  pip install langchain-openai   then set OPENAI_API_KEY
"""
import os
import sqlite3
import sys
from langchain_core.messages import AIMessageChunk, HumanMessage, trim_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, MessagesState, StateGraph

# ── 1. CONFIG ─────────────────────────────────────────────────────────────
HF_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"        # local default (same model as EP48)
API_MODEL = "gpt-4o-mini"                      # used when OPENAI_API_KEY is set
SYSTEM = "You are Nova, a friendly assistant. Be concise. Use what the user told you earlier."
DB_FILE = "memory.db"                          # the chat lives here, on disk
WINDOW = 10                                    # messages sent to the model each turn


# ── 2. MODEL: an API model if a key is set, else a local Hugging Face model ─
def get_llm():
    if os.getenv("OPENAI_API_KEY"):
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=API_MODEL, temperature=0.7)
    from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
    pipe = HuggingFacePipeline.from_model_id(
        HF_MODEL, task="text-generation",
        pipeline_kwargs=dict(max_new_tokens=256, do_sample=True, temperature=0.7,
                             top_p=0.9, return_full_text=False))
    return ChatHuggingFace(llm=pipe)


# ── 3. MEMORY: save EVERYTHING to disk, send only a window to the model ───
def get_checkpointer():
    return SqliteSaver(sqlite3.connect(DB_FILE, check_same_thread=False))


window = trim_messages(max_tokens=WINDOW, token_counter=len,      # counts messages
                       strategy="last", start_on="human", include_system=False)


# ── 4. CHAIN + GRAPH: prompt | model, one node, a checkpointer for memory ──
def build_bot(llm, checkpointer):
    chain = ChatPromptTemplate.from_messages(
        [("system", SYSTEM), MessagesPlaceholder("history")]) | llm

    def call_model(state):
        return {"messages": [chain.invoke({"history": window.invoke(state["messages"])})]}

    graph = StateGraph(MessagesState)
    graph.add_node("model", call_model)
    graph.add_edge(START, "model")
    return graph.compile(checkpointer=checkpointer)


def stream_reply(bot, session_id, text):
    """Yield the growing reply so the UI shows it token by token."""
    out, cfg = "", {"configurable": {"thread_id": session_id}}
    for chunk, _ in bot.stream({"messages": [HumanMessage(text)]}, cfg, stream_mode="messages"):
        if isinstance(chunk, AIMessageChunk) and chunk.content:
            out += chunk.content
            yield out


# ── 5. SERVE: a Gradio web UI or a terminal loop ──────────────────────────
def run_ui(bot):
    import gradio as gr
    gr.ChatInterface(
        lambda msg, hist, sid: stream_reply(bot, sid or "demo", msg),
        additional_inputs=[gr.Textbox("demo", label="session id (same id = same memory)")],
        title="Nova - a chatbot with memory").launch()


def run_cli(bot, sid="demo"):
    print(f"session '{sid}' - type /forget to wipe it, quit to stop")
    while (text := input("you > ").strip()) != "quit":
        if text == "/forget":
            bot.checkpointer.delete_thread(sid)
            print("nova > memory wiped")
            continue
        reply = ""
        for reply in stream_reply(bot, sid, text):
            pass
        print("nova >", reply)


if __name__ == "__main__":
    nova = build_bot(get_llm(), get_checkpointer())
    run_cli(nova) if "--cli" in sys.argv else run_ui(nova)
