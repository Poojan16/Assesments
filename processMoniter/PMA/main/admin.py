from django.contrib import admin
from .models import Machine, Snapshot, ProcessRow


@admin.register(Machine)
class MachineAdmin(admin.ModelAdmin):
    list_display = ("hostname", "created_at")
    search_fields = ("hostname",)


@admin.register(Snapshot)
class SnapshotAdmin(admin.ModelAdmin):
    list_display = ("machine", "created_at", "process_count")
    list_filter = ("machine",)


@admin.register(ProcessRow)
class ProcessRowAdmin(admin.ModelAdmin):
    list_display = ("snapshot", "pid", "ppid", "name", "cpu_percent", "memory_rss", "memory_percent")
    list_filter = ("snapshot",)
    search_fields = ("name",)
