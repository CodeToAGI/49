# EP49 — Build a Chatbot with Memory (LangChain + LangGraph)

**Deep Learning Series · Episode 49 · Module 9 (Project)**

A complete multi-turn chatbot with persistent memory in only **93 lines**.

## What This Project Does

- Remembers conversations across restarts (SQLite)
- Streams replies token-by-token
- Works with a local Hugging Face model **or** an OpenAI API model
- Clean Gradio chat interface + terminal mode
- Windowed memory so cost stays low

## Quick Start

```bash
pip install langchain-core langchain-huggingface transformers torch gradio
pip install langgraph langgraph-checkpoint-sqlite

# Optional (for stronger model)
pip install langchain-openai
# then set OPENAI_API_KEY

python ep49_chatbot.py          # opens Gradio UI
python ep49_chatbot.py --cli    # terminal chat
