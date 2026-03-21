"""run_pipeline.py — management command to trigger the full pipeline for a file."""

import logging
from pathlib import Path

from django.core.management.base import BaseCommand, CommandError

from dashboard.models import ProcessedFile
from dashboard.pipeline_runner import run_pipeline_for_file

log = logging.getLogger(__name__)


class Command(BaseCommand):
    """Django management command: ``python manage.py run_pipeline --file <path>``."""

    help = "Run the full ingestion pipeline for a given .xlsx file."

    def add_arguments(self, parser) -> None:
        """Register the ``--file`` argument.

        Args:
            parser: argparse parser provided by Django.
        """
        parser.add_argument(
            "--file",
            type=str,
            required=True,
            help="Path to the .xlsx file to process.",
        )

    def handle(self, *args, **options) -> None:
        """Execute the pipeline and update the ProcessedFile record.

        Args:
            *args: Positional arguments (unused).
            **options: Parsed options dict containing ``file``.

        Raises:
            CommandError: If the file is not found or the pipeline fails.
        """
        filepath = Path(options["file"])
        filename = filepath.name

        if not filepath.exists():
            raise CommandError(f"File not found: {filepath}")

        record, _ = ProcessedFile.objects.get_or_create(filename=filename)
        record.status = ProcessedFile.Status.PENDING
        record.error_message = None
        record.save(update_fields=["status", "error_message"])

        try:
            row_count = run_pipeline_for_file(filepath)
            record.status = ProcessedFile.Status.SUCCESS
            record.row_count = row_count
            record.error_message = None
            self.stdout.write(
                self.style.SUCCESS(
                    f"Pipeline succeeded for '{filename}' — {row_count} rows."
                )
            )
            log.info("Pipeline succeeded for '%s' — %d rows.", filename, row_count)
        except Exception as exc:
            record.status = ProcessedFile.Status.FAILED
            record.error_message = str(exc)
            log.error("Pipeline failed for '%s': %s", filename, exc)
            raise CommandError(f"Pipeline failed: {exc}") from exc
        finally:
            record.save(update_fields=["status", "row_count", "error_message"])
