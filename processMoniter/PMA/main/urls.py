from django.urls import path
from . import views


urlpatterns = [
    path("", views.dashboard, name="dashboard"),
    path("api/ingest/", views.IngestView.as_view(), name="api-ingest"),
    path("api/latest/<str:hostname>/", views.LatestSnapshotView.as_view(), name="api-latest"),
    path("api/machines/", views.MachinesView.as_view(), name="api-machines"),
] 