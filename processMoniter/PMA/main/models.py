from django.db import models


class Machine(models.Model):
    hostname = models.CharField(max_length=255, unique=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self) -> str:
        return self.hostname


class Snapshot(models.Model):
    machine = models.ForeignKey(Machine, on_delete=models.CASCADE, related_name="snapshots")
    created_at = models.DateTimeField(auto_now_add=True)
    process_count = models.PositiveIntegerField(default=0)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"Snapshot({self.machine.hostname} @ {self.created_at:%Y-%m-%d %H:%M:%S})"


class ProcessRow(models.Model):
    snapshot = models.ForeignKey(Snapshot, on_delete=models.CASCADE, related_name="process_rows")
    pid = models.IntegerField()
    ppid = models.IntegerField()
    name = models.CharField(max_length=255)
    cpu_percent = models.FloatField()
    memory_rss = models.BigIntegerField()
    memory_percent = models.FloatField()

    class Meta:
        indexes = [
            models.Index(fields=["snapshot", "pid"]),
            models.Index(fields=["snapshot", "ppid"]),
        ]

    def __str__(self) -> str:
        return f"{self.name} (pid={self.pid}, ppid={self.ppid})"
