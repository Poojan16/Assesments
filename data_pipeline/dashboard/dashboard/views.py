"""views.py — dashboard list view and pipeline trigger endpoint."""

import logging
import sys
from pathlib import Path

from django.http import HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_GET, require_POST

from .models import ProcessedFile

log = logging.getLogger(__name__)

_INCOMING_DIR = Path(__file__).resolve().parent.parent.parent / "tmp" / "incoming"

# Ensure data_pipeline/ root is on sys.path for pg_store import
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _top_accounts(top_n: int = 3) -> list[dict]:
    """Query PostgreSQL for the top *top_n* accounts by total amount.

    Args:
        top_n: Number of accounts to return.

    Returns:
        List of dicts with keys ``account_id`` and ``total_amount``,
        sorted descending. Returns an empty list on any DB error.
    """
    try:
        from persistence.pg_store import get_connection
        conn = get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT account_id, SUM(amount) AS total_amount
                    FROM transactions
                    GROUP BY account_id
                    ORDER BY total_amount DESC
                    LIMIT %s;
                    """,
                    (top_n,),
                )
                rows = cur.fetchall()
        finally:
            conn.close()
        return [{"account_id": r[0], "total_amount": round(r[1], 2)} for r in rows]
    except Exception as exc:
        log.warning("Could not fetch top accounts: %s", exc)
        return []


@require_GET
def dashboard_list(request: HttpRequest) -> HttpResponse:
    """Render the dashboard overview page.

    Lists all :class:`~dashboard.models.ProcessedFile` records newest-first.
    For the most recent success record, fetches the top 3 accounts by total
    amount from PostgreSQL via :func:`_top_accounts`.

    Args:
        request: Incoming GET request.

    Returns:
        Rendered HTML response.
    """
    records = ProcessedFile.objects.all()
    top_accounts: list[dict] = []

    success_record = records.filter(status=ProcessedFile.Status.SUCCESS).first()
    if success_record:
        top_accounts = _top_accounts(top_n=3)

    return render(
        request,
        "dashboard/index.html",
        {"records": records, "top_accounts": top_accounts},
    )


@require_GET
def insights(request: HttpRequest) -> JsonResponse:
    """Run the LangGraph agent and return a markdown insights summary.

    Calls :func:`ai.langchain_agent.run_agent` with the default spike/anomaly
    question and returns the markdown as JSON.

    Args:
        request: Incoming GET request.

    Returns:
        JSON ``{"insights": "<markdown>"}`` or
        ``{"error": "<message>"}`` with HTTP 500 on failure.
    """
    try:
        from ai.langchain_agent import run_agent
        markdown = run_agent(
            "Which accounts show the highest 7-day rolling spike? "
            "Are there any anomalies?"
        )
        return JsonResponse({"insights": markdown})
    except Exception as exc:
        log.error("Insights agent failed: %s", exc)
        return JsonResponse({"error": str(exc)}, status=500)


@require_POST
def trigger_pipeline(request: HttpRequest) -> JsonResponse:
    """Trigger the full pipeline for a file in the incoming directory.

    Expects ``?file=<filename>`` query parameter.
    Creates or updates a :class:`~dashboard.models.ProcessedFile` record.

    Args:
        request: Incoming POST request.

    Returns:
        JSON ``{"status": "queued", "file": "<filename>"}`` on success,
        or HTTP 404 JSON if the file is not found.
    """
    from .pipeline_runner import run_pipeline_for_file

    filename = request.GET.get("file", "").strip()
    if not filename:
        return JsonResponse({"error": "Missing 'file' query parameter."}, status=400)

    filepath = _INCOMING_DIR / filename
    if not filepath.exists():
        log.warning("Trigger requested for missing file: %s", filepath)
        return JsonResponse({"error": f"File not found: {filename}"}, status=404)

    record, _ = ProcessedFile.objects.get_or_create(filename=filename)
    record.status = ProcessedFile.Status.PENDING
    record.error_message = None
    record.save(update_fields=["status", "error_message"])

    try:
        row_count = run_pipeline_for_file(filepath)
        record.status = ProcessedFile.Status.SUCCESS
        record.row_count = row_count
        record.error_message = None
        log.info("Pipeline succeeded for '%s' — %d rows.", filename, row_count)
    except Exception as exc:
        record.status = ProcessedFile.Status.FAILED
        record.error_message = str(exc)
        log.error("Pipeline failed for '%s': %s", filename, exc)
    finally:
        record.save(update_fields=["status", "row_count", "error_message"])

    return JsonResponse({"status": record.status, "file": filename})
