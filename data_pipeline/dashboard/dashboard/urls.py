"""urls.py — URL configuration for the dashboard app."""

from django.urls import path
from . import views

urlpatterns = [
    path("", views.dashboard_list, name="dashboard-list"),
    path("trigger/", views.trigger_pipeline, name="trigger-pipeline"),
    path("insights/", views.insights, name="insights"),
]
