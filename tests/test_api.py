import base64
import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("httpx")
from fastapi.testclient import TestClient

models_ready = (Path(__file__).resolve().parents[1] / "models" / "lumanet_fast_int8.onnx").exists()
pytestmark = pytest.mark.skipif(not models_ready, reason="exported models not available")


@pytest.fixture(scope="module")
def client(monkeypatch_module=None):
    from api.index import app

    return TestClient(app)


def upload_payload():
    array = (np.random.default_rng(0).random((96, 128, 3)) * 255).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, "PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def test_health_reports_database_state(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["database"] in {"connected", "not_configured", "unreachable"}


def test_sample_processing_returns_all_arms_and_metrics(client):
    sample = client.get("/api/samples").json()[0]["id"]
    body = client.post("/api/process", json={"sample_id": sample}).json()
    assert set(body["metrics"]) == {"traditional", "learned", "hybrid"}
    assert all(body["images"][arm] for arm in ("traditional", "learned", "hybrid", "reference"))
    assert body["simulated"] is False


def test_degradation_slows_nothing_and_changes_route_inputs(client):
    sample = client.get("/api/samples").json()[0]["id"]
    clean = client.post("/api/process", json={"sample_id": sample}).json()
    dark = client.post("/api/process", json={"sample_id": sample,
                                             "degradation": {"exposure_stops": 3, "noise": 0.5}}).json()
    assert dark["controls"]["exposure"] == 8.0
    assert dark["stats"]["sigma"] != clean["stats"]["sigma"]


def test_upload_is_marked_as_simulated(client):
    body = client.post("/api/process", json={"image_base64": upload_payload()}).json()
    assert body["simulated"] is True
    assert body["metrics"] is None and body["images"]["reference"] is None


@pytest.mark.parametrize(
    "payload,status",
    [
        ({}, 400),
        ({"sample_id": "../../etc/passwd"}, 400),
        ({"sample_id": "scene-9999"}, 404),
        ({"image_base64": "not-an-image"}, 400),
        ({"sample_id": "x", "mode": "turbo"}, 422),
    ],
)
def test_bad_requests_are_rejected(client, payload, status):
    assert client.post("/api/process", json=payload).status_code == status


def test_telemetry_shape_without_database(client):
    body = client.get("/api/telemetry").json()
    assert {"database", "total", "by_route", "recent"} <= set(body)
