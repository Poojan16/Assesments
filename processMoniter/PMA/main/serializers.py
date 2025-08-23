from rest_framework import serializers
from .models import ProcessRow, Snapshot


class ProcessRowSerializer(serializers.ModelSerializer):
    class Meta:
        model = ProcessRow
        fields = [
            "pid",
            "ppid",
            "name",
            "cpu_percent",
            "memory_rss",
            "memory_percent",
        ]


class SnapshotSerializer(serializers.ModelSerializer):
    processes = serializers.SerializerMethodField()

    class Meta:
        model = Snapshot
        fields = ["id", "created_at", "process_count", "processes"]

    def get_processes(self, obj: Snapshot):
        rows = obj.process_rows.all().only(
            "pid", "ppid", "name", "cpu_percent", "memory_rss", "memory_percent"
        )
        return ProcessRowSerializer(rows, many=True).data 