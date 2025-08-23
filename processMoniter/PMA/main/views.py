from django.conf import settings
from django.shortcuts import render
from django.utils import timezone

from rest_framework.authentication import BaseAuthentication
from rest_framework.exceptions import AuthenticationFailed
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from rest_framework.views import APIView

from .models import Machine, Snapshot, ProcessRow
from .serializers import SnapshotSerializer


class ApiKeyAuthentication(BaseAuthentication):
    def authenticate(self, request):
        api_key = request.headers.get("X-API-KEY")
        if not api_key or api_key != settings.API_KEY:
            raise AuthenticationFailed("Invalid or missing API key")
        return (None, None)


class IngestView(APIView):
    authentication_classes = [ApiKeyAuthentication]
    permission_classes = [AllowAny]

    def post(self, request):
        data = request.data or {}
        hostname = data.get("hostname")
        processes = data.get("processes") or []

        if not hostname or not isinstance(processes, list):
            return Response({"detail": "hostname and processes are required"}, status=400)

        machine, _ = Machine.objects.get_or_create(hostname=hostname)
        snapshot = Snapshot.objects.create(machine=machine, created_at=timezone.now())

        process_rows = []
        for proc in processes:
            try:
                process_rows.append(
                    ProcessRow(
                        snapshot=snapshot,
                        pid=int(proc.get("pid", 0)),
                        ppid=int(proc.get("ppid", 0)),
                        name=str(proc.get("name", ""))[:255],
                        cpu_percent=float(proc.get("cpu_percent", 0.0)),
                        memory_rss=int(proc.get("memory_rss", 0)),
                        memory_percent=float(proc.get("memory_percent", 0.0)),
                    )
                )
            except Exception:
                continue

        if process_rows:
            ProcessRow.objects.bulk_create(process_rows, batch_size=1000)
            Snapshot.objects.filter(id=snapshot.id).update(process_count=len(process_rows))

        return Response({"status": "ok", "snapshot_id": snapshot.id, "process_count": len(process_rows)})


class LatestSnapshotView(APIView):
    permission_classes = [AllowAny]

    def get(self, request, hostname: str):
        try:
            machine = Machine.objects.get(hostname=hostname)
        except Machine.DoesNotExist:
            return Response({"detail": "machine not found"}, status=404)

        snapshot = machine.snapshots.order_by("-created_at").first()
        if not snapshot:
            return Response({"detail": "no snapshots for machine"}, status=404)

        serializer = SnapshotSerializer(snapshot)
        return Response({
            "hostname": machine.hostname,
            "snapshot_time": snapshot.created_at,
            "processes": serializer.data.get("processes", []),
        })


class MachinesView(APIView):
    permission_classes = [AllowAny]

    def get(self, request):
        hosts = list(Machine.objects.values_list("hostname", flat=True).order_by("hostname"))
        return Response({"machines": hosts})


def dashboard(request):
    machines = Machine.objects.all().order_by("hostname")
    return render(request, "main/dashboard.html", {"machines": machines})
