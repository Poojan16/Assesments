"""state.py — TypedDict state schema for the anomaly-detection LangGraph pipeline.

Every node receives the full state and returns a *partial* dict containing only
the keys it mutates.  LangGraph merges the partial update back into the state
automatically, so nodes never need to copy keys they did not touch.
"""

from typing import Any, TypedDict


class AnomalyState(TypedDict):
    """Typed state carried through every node of the anomaly-detection graph.

    Attributes:
        source_path:        Absolute path (or connection string) to the input
                            file.  This is the only value supplied by the
                            caller; every other field starts empty/None.
        raw_data:           The pandas DataFrame loaded by ``load_data``.
                            Stored as ``Any`` because TypedDict cannot express
                            a DataFrame type without importing pandas at the
                            schema level.
        validation_errors:  List of human-readable strings describing schema
                            or dtype problems found by ``validate_schema``.
                            Empty list means the data passed all checks.
        stats:              Dict produced by ``compute_stats`` containing
                            per-column null counts, numeric outlier flags,
                            duplicate counts, invalid-value sets, and
                            descriptive statistics.
        llm_response:       Raw string returned by the LLM node before
                            Pydantic parsing.
        final_output:       Validated Pydantic model (serialised to a plain
                            dict by ``format_output``) that is the graph's
                            public result.
    """

    source_path: str
    raw_data: Any                        # pandas.DataFrame at runtime
    validation_errors: list[str]
    stats: dict[str, Any]
    llm_response: str
    final_output: dict[str, Any]
