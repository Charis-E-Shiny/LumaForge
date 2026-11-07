"""LumaForge API: a FastAPI app that Vercel runs as one Python serverless function."""
from __future__ import annotations

import base64
import hashlib
import io
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from PIL import Image, ImageOps
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from api import store
from lumaforge.degrade import add_sensor_noise, colour_cast, low_light
from lumaforge.isp.pipeline import ccm_for
from lumaforge.metrics_lite import delta_e_76, psnr
from lumaforge.runtime import render
from lumaforge.unprocess import unprocess

app = FastAPI(title="LumaForge API", version="0.1.0", docs_url="/api/docs", openapi_url="/api/openapi.json")

SAMPLE_DIR = ROOT / "samples"
RESULTS_PATH = ROOT / "results" / "summary.json"
MAX_UPLOAD_SIDE = 768
RAW_SCALE = 30000.0


class Degradation(BaseModel):
    """Live sensor simulation applied in the RAW domain before processing."""

    exposure_stops: float = Field(0.0, ge=0.0, le=4.0)
    noise: float = Field(0.0, ge=0.0, le=1.0)
    cast: float = Field(0.0, ge=-1.0, le=1.0)


class ProcessRequest(BaseModel):
    sample_id: str | None = None
    image_base64: str | None = None
    mode: str = Field("auto", pattern="^(auto|fast|quality)$")
    degradation: Degradation = Degradation()


def encode(image: np.ndarray, quality: int = 90) -> str:
    buffer = io.BytesIO()
    Image.fromarray((np.clip(image, 0, 1) * 255 + 0.5).astype(np.uint8)).save(buffer, "JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def load_sample(sample_id: str) -> tuple[np.ndarray, np.ndarray, str, dict]:
    if not sample_id.replace("-", "").replace("_", "").isalnum():
        raise HTTPException(400, "invalid sample id")
    path = SAMPLE_DIR / f"{sample_id}.npz"
    if not path.is_file():
        raise HTTPException(404, "unknown sample")
    data = np.load(path)
    info = json.loads(str(data["info"]))
    return (
        data["bayer"].astype(np.float32) / RAW_SCALE,
        data["rgb"].astype(np.float32) / 255.0,
        info["camera"],
        info,
    )


def decode_upload(payload: str) -> np.ndarray:
    try:
        raw = base64.b64decode(payload.split(",")[-1], validate=True)
        image = ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB")
    except Exception as exc:
        raise HTTPException(400, "could not decode image") from exc
    image.thumbnail((MAX_UPLOAD_SIDE, MAX_UPLOAD_SIDE))
    return np.asarray(image, dtype=np.float32) / 255.0


def degrade(bayer: np.ndarray, settings: Degradation, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = bayer
    if settings.exposure_stops > 0:
        out = low_light(out, 2.0 ** -settings.exposure_stops, rng)
    if settings.noise > 0:
        out = add_sensor_noise(out, 5e-3 * settings.noise, 3e-5 * settings.noise, rng)
    if settings.cast != 0:
        strength = abs(settings.cast) * 0.5
        warm = settings.cast > 0
        out = colour_cast(out, np.exp(strength if warm else -strength), np.exp(-strength if warm else strength))
    return out


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "database": store.status(), "version": app.version}


@app.get("/api/samples")
def samples() -> list[dict]:
    index = SAMPLE_DIR / "index.json"
    return json.loads(index.read_text()) if index.exists() else []


@app.post("/api/process")
def process(request: ProcessRequest) -> dict:
    started = time.perf_counter()
    reference = None
    simulated = False
    if request.sample_id:
        bayer, reference, camera, info = load_sample(request.sample_id)
        source = request.sample_id
    elif request.image_base64:
        rgb = decode_upload(request.image_base64)
        camera, info, simulated = None, {"camera": "simulated"}, True
        bayer = unprocess(rgb, ccm_for(None))
        source = "upload:" + hashlib.sha256(rgb.tobytes()).hexdigest()[:12]
    else:
        raise HTTPException(400, "send sample_id or image_base64")

    seed = int(hashlib.sha256(source.encode()).hexdigest()[:8], 16)
    degraded = degrade(bayer, request.degradation, seed)
    # The phone knows how far it under-exposed, so that gain is passed as metadata.
    rendered = render(degraded, camera, request.mode, compensation=2.0**request.degradation.exposure_stops)
    height, width = rendered.hybrid.shape[:2]

    metrics = None
    if reference is not None:
        target = reference[: height, : width]
        arms = {"traditional": rendered.traditional, "learned": rendered.learned, "hybrid": rendered.hybrid}
        metrics = {
            name: {
                "psnr": round(psnr(image[: height, : width], target), 2),
                "delta_e": round(delta_e_76(image[: height, : width], target), 2),
            }
            for name, image in arms.items()
        }
    total = round((time.perf_counter() - started) * 1000, 1)
    document = {
        "createdAt": datetime.now(timezone.utc),
        "source": "sample" if request.sample_id else "upload",
        "sample": request.sample_id,
        "camera": info.get("camera"),
        "degradation": request.degradation.model_dump(),
        "route": rendered.route,
        "stats": rendered.stats,
        "timings_ms": rendered.timings_ms,
        "total_ms": total,
        "metrics": metrics,
    }
    saved = store.log_run(document)
    return {
        "images": {
            "traditional": encode(rendered.traditional[: height, : width]),
            "learned": encode(rendered.learned[: height, : width]),
            "hybrid": encode(rendered.hybrid),
            "reference": encode(reference[: height, : width]) if reference is not None else None,
        },
        "metrics": metrics,
        "route": rendered.route,
        "model": rendered.model,
        "stats": rendered.stats,
        "controls": rendered.controls,
        "gains": rendered.gains,
        "timings_ms": rendered.timings_ms,
        "total_ms": total,
        "simulated": simulated,
        "logged": saved,
        "info": info,
    }


@app.get("/api/results")
def results() -> dict:
    """Published evaluation. MongoDB holds the same document once it has been seeded."""
    document = store.latest_results()
    if document:
        return document
    if RESULTS_PATH.exists():
        return json.loads(RESULTS_PATH.read_text())
    raise HTTPException(404, "results not generated yet")


@app.get("/api/telemetry")
def telemetry() -> dict:
    return store.telemetry()
