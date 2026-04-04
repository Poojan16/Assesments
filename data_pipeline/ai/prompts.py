"""prompts.py — all LangChain prompt templates for the anomaly-detection pipeline.

Every template is a module-level named constant so nodes import the object
directly rather than constructing prompts inline.  No f-string prompt
construction appears anywhere in the codebase.
"""

from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

# ── System persona ─────────────────────────────────────────────────────────────
_SYSTEM_TEMPLATE = SystemMessagePromptTemplate.from_template(
    "You are a senior data-quality engineer specialising in financial transaction "
    "pipelines. Your job is to analyse ingested data and produce a precise, "
    "structured anomaly report. Be concise, data-driven, and avoid generic filler. "
    "Always ground every finding in the statistics provided."
)

# ── Main anomaly-analysis prompt ───────────────────────────────────────────────
# Input variables:
#   {validation_errors}  — newline-separated schema/dtype errors (may be "None")
#   {stats_json}         — JSON string of the compute_stats output
#   {row_count}          — total rows in the ingested DataFrame
#   {column_list}        — comma-separated column names
ANOMALY_ANALYSIS_PROMPT = ChatPromptTemplate.from_messages(
    [
        _SYSTEM_TEMPLATE,
        HumanMessagePromptTemplate.from_template(
            "## Ingested dataset overview\n"
            "- Total rows  : {row_count}\n"
            "- Columns     : {column_list}\n\n"
            "## Schema / dtype validation errors\n"
            "{validation_errors}\n\n"
            "## Computed statistics (JSON)\n"
            "```json\n{stats_json}\n```\n\n"
            "## Task\n"
            "Analyse the statistics above and produce a structured anomaly report "
            "that directly answers: **'Summarise anomalies in the ingested data "
            "(flag outliers, missing fields, unexpected values).'**\n\n"
            "Your response MUST be valid JSON matching this exact schema:\n"
            "{{\n"
            '  "summary": "<one-sentence executive summary>",\n'
            '  "outliers": [\n'
            '    {{"column": "<col>", "description": "<what was found>", "affected_rows": <int>}}\n'
            "  ],\n"
            '  "missing_fields": [\n'
            '    {{"column": "<col>", "null_count": <int>, "null_pct": <float>}}\n'
            "  ],\n"
            '  "unexpected_values": [\n'
            '    {{"column": "<col>", "description": "<what was found>", "examples": ["<val>", ...]}}\n'
            "  ],\n"
            '  "severity": "low" | "medium" | "high",\n'
            '  "recommendations": ["<actionable fix>", ...]\n'
            "}}\n\n"
            "Return ONLY the JSON object — no markdown fences, no prose outside the JSON."
        ),
    ]
)

# ── Retry / repair prompt (used when the first LLM response is not valid JSON) ─
# Input variables:
#   {bad_response}  — the malformed LLM output from the first attempt
REPAIR_PROMPT = ChatPromptTemplate.from_messages(
    [
        _SYSTEM_TEMPLATE,
        HumanMessagePromptTemplate.from_template(
            "The following response was supposed to be a JSON object matching the "
            "anomaly report schema but it is malformed or contains extra text:\n\n"
            "{bad_response}\n\n"
            "Return ONLY the corrected, valid JSON object — no markdown fences, "
            "no prose, no explanation."
        ),
    ]
)
