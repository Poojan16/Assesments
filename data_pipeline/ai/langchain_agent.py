"""langchain_agent.py — LangGraph analytics agent with 3 nodes.

Nodes
-----
1. data_loader  — fetch top-account metrics from Postgres or a Polars result
2. analyst      — LLM answers the spike/anomaly question via PromptTemplate
3. formatter    — renders the answer as a clean markdown summary

CLI usage
---------
    python ai/langchain_agent.py --question "Which accounts spiked this week?"

Django endpoint
---------------
    GET /dashboard/insights/   →  {"insights": "<markdown>"}
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, TypedDict

from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langgraph.graph import END, StateGraph

load_dotenv(Path(__file__).parent.parent / ".env")

# Ensure data_pipeline/ root is importable
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)

# ── Shared multi-turn memory (CLI) ────────────────────────────────────────────
_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)


# ── Graph state ───────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    """Typed state passed between LangGraph nodes.

    Attributes:
        question:    The user's natural-language question.
        metrics:     Serialised metrics string loaded by data_loader.
        raw_answer:  LLM response text produced by analyst.
        markdown:    Final formatted markdown produced by formatter.
    """

    question: str
    metrics: str
    raw_answer: str
    markdown: str


# ── LLM factory ───────────────────────────────────────────────────────────────

def _build_llm() -> Any:
    """Instantiate the LLM based on ``LANGCHAIN_LLM_PROVIDER`` env var.

    Supports ``"openai"`` (default) and ``"anthropic"``.

    Returns:
        A LangChain chat model instance.

    Raises:
        ValueError: If the provider is not recognised.
    """
    provider = os.environ.get("LANGCHAIN_LLM_PROVIDER", "openai").lower()
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.2,
            api_key=os.environ.get("OPENAI_API_KEY"),
        )
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic  # type: ignore[import]
        return ChatAnthropic(model="claude-3-haiku-20240307", temperature=0.2)
    raise ValueError(f"Unsupported LANGCHAIN_LLM_PROVIDER: {provider!r}")


# ── Node 1 — data_loader ──────────────────────────────────────────────────────

def _load_metrics_from_pg() -> str:
    """Query Postgres for the top 10 accounts by max rolling_7d_sum.

    Returns:
        A newline-separated string of ``account_id | rolling_7d_sum`` rows,
        or an error message string if the query fails.
    """
    try:
        from persistence.pg_store import get_connection
        conn = get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT account_id, SUM(amount) AS total
                    FROM transactions
                    GROUP BY account_id
                    ORDER BY total DESC
                    LIMIT 10;
                    """
                )
                rows = cur.fetchall()
        finally:
            conn.close()
        lines = [f"{r[0]} | {round(r[1], 2)}" for r in rows]
        return "account_id | total_amount_usd\n" + "\n".join(lines)
    except Exception as exc:
        log.warning("Postgres unavailable, falling back to empty metrics: %s", exc)
        return "No data available."


def node_data_loader(state: AgentState) -> AgentState:
    """LangGraph node 1 — load aggregated account metrics.

    Attempts to query Postgres; falls back to a placeholder string if
    the database is unreachable.

    Args:
        state: Current graph state.

    Returns:
        Updated state with ``metrics`` populated.
    """
    log.info("Node data_loader: fetching metrics.")
    metrics = _load_metrics_from_pg()
    log.info("Metrics loaded (%d chars).", len(metrics))
    return {**state, "metrics": metrics}


# ── Node 2 — analyst ──────────────────────────────────────────────────────────

_ANALYST_PROMPT = PromptTemplate(
    input_variables=["chat_history", "metrics", "question"],
    template=(
        "You are a financial data analyst.\n\n"
        "Conversation so far:\n{chat_history}\n\n"
        "Here are the top accounts by total transaction amount (USD):\n"
        "{metrics}\n\n"
        "Question: {question}\n\n"
        "Identify which accounts show the highest 7-day rolling spike "
        "and flag any anomalies. Be concise and data-driven."
    ),
)


def node_analyst(state: AgentState) -> AgentState:
    """LangGraph node 2 — run the LLM analyst over the loaded metrics.

    Injects conversation history from :data:`_memory` for multi-turn support.

    Args:
        state: Current graph state (must have ``metrics`` and ``question``).

    Returns:
        Updated state with ``raw_answer`` populated.
    """
    log.info("Node analyst: invoking LLM.")
    llm = _build_llm()
    chain = _ANALYST_PROMPT | llm | StrOutputParser()

    chat_history = _memory.load_memory_variables({}).get("chat_history", "")

    raw_answer: str = chain.invoke(
        {
            "chat_history": chat_history,
            "metrics": state["metrics"],
            "question": state["question"],
        }
    )
    # Persist turn to memory
    _memory.save_context(
        {"input": state["question"]},
        {"output": raw_answer},
    )
    log.info("LLM response received (%d chars).", len(raw_answer))
    return {**state, "raw_answer": raw_answer}


# ── Node 3 — formatter ────────────────────────────────────────────────────────

_FORMATTER_PROMPT = PromptTemplate(
    input_variables=["raw_answer"],
    template=(
        "Reformat the following analysis as a clean markdown report.\n"
        "Use headers, bullet points, and bold text where appropriate.\n"
        "Do not add new information — only reformat.\n\n"
        "{raw_answer}"
    ),
)


def node_formatter(state: AgentState) -> AgentState:
    """LangGraph node 3 — reformat the raw LLM answer as markdown.

    Args:
        state: Current graph state (must have ``raw_answer``).

    Returns:
        Updated state with ``markdown`` populated.
    """
    log.info("Node formatter: rendering markdown.")
    llm = _build_llm()
    chain = _FORMATTER_PROMPT | llm | StrOutputParser()
    markdown: str = chain.invoke({"raw_answer": state["raw_answer"]})
    log.info("Markdown formatted (%d chars).", len(markdown))
    return {**state, "markdown": markdown}


# ── Graph assembly ────────────────────────────────────────────────────────────

def _build_graph() -> Any:
    """Assemble and compile the 3-node LangGraph StateGraph.

    Returns:
        A compiled LangGraph runnable.
    """
    builder: StateGraph = StateGraph(AgentState)

    builder.add_node("data_loader", node_data_loader)
    builder.add_node("analyst", node_analyst)
    builder.add_node("formatter", node_formatter)

    builder.set_entry_point("data_loader")
    builder.add_edge("data_loader", "analyst")
    builder.add_edge("analyst", "formatter")
    builder.add_edge("formatter", END)

    return builder.compile()


# ── Public API ────────────────────────────────────────────────────────────────

def run_agent(question: str) -> str:
    """Run the full LangGraph pipeline for *question* and return markdown.

    Args:
        question: Natural-language question to answer.

    Returns:
        Markdown-formatted insights string.
    """
    graph = _build_graph()
    initial_state: AgentState = {
        "question": question,
        "metrics": "",
        "raw_answer": "",
        "markdown": "",
    }
    final_state: AgentState = graph.invoke(initial_state)
    return final_state["markdown"]


# ── CLI entry point ───────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Returns:
        Namespace with ``question`` string.
    """
    parser = argparse.ArgumentParser(description="Run the LangGraph analytics agent.")
    parser.add_argument(
        "--question",
        type=str,
        default="Which accounts show the highest 7-day rolling spike? Are there any anomalies?",
        help="Question to ask the agent.",
    )
    return parser.parse_args()


def main() -> None:
    """Multi-turn CLI loop: ask questions, print markdown, repeat until EOF."""
    args = _parse_args()

    # First turn from --question flag
    print("\n" + run_agent(args.question))

    # Subsequent turns
    while True:
        try:
            follow_up = input("\nFollow-up question (Ctrl-C to exit): ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break
        if not follow_up:
            continue
        print("\n" + run_agent(follow_up))


if __name__ == "__main__":
    main()
