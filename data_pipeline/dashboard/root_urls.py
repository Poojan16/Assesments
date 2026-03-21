"""root urls.py — project-level URL routing."""

from django.urls import include, path

urlpatterns = [
    path("dashboard/", include("dashboard.urls")),
]
