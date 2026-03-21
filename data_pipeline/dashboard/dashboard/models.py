"""models.py — ORM models for the dashboard app."""

from django.db import models


class ProcessedFile(models.Model):
    """Record of a file that has passed through the ingestion pipeline.

    Attributes:
        filename:      Original filename as received from the watcher.
        processed_at:  Timestamp set automatically on creation.
        row_count:     Number of rows successfully ingested.
        status:        One of ``pending``, ``success``, or ``failed``.
        error_message: Populated only when ``status == "failed"``.
    """

    class Status(models.TextChoices):
        PENDING = "pending", "Pending"
        SUCCESS = "success", "Success"
        FAILED = "failed", "Failed"

    filename = models.CharField(max_length=512)
    processed_at = models.DateTimeField(auto_now_add=True)
    row_count = models.IntegerField(default=0)
    status = models.CharField(
        max_length=16,
        choices=Status.choices,
        default=Status.PENDING,
    )
    error_message = models.TextField(null=True, blank=True)

    class Meta:
        ordering = ["-processed_at"]

    def __str__(self) -> str:
        return f"{self.filename} [{self.status}]"
