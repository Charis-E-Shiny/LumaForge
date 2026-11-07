# API Reference

Base path: `/api`. Interactive documentation is served at `/api/docs`.

## `GET /api/health`

```json
{ "status": "ok", "database": "connected", "version": "0.1.0" }
```

`database` is `connected`, `not_configured` or `unreachable`.

## `GET /api/samples`

Lists the demo captures.

```json
[{ "id": "scene-0020", "title": "Scene 0020", "camera": "Google Pixel", "iso": "unknown", "thumbnail": "/samples/scene-0020.jpg" }]
```

## `POST /api/process`

Runs the traditional, learned and hybrid pipelines.

Send either `sample_id` or `image_base64`.

| Field | Type | Default | Description |
|---|---|---|---|
| `sample_id` | string | | Demo capture |
| `image_base64` | string | | Image data URL, converted to a simulated mosaic |
| `mode` | string | `auto` | `auto`, `fast` or `quality` |
| `degradation.exposure_stops` | number | `0` | 0 to 4 |
| `degradation.noise` | number | `0` | 0 to 1 |
| `degradation.cast` | number | `0` | -1 (cool) to 1 (warm) |

```bash
curl -X POST http://localhost:8000/api/process \
  -H "Content-Type: application/json" \
  -d '{"sample_id":"scene-0020","degradation":{"exposure_stops":2}}'
```

Response fields:

| Field | Description |
|---|---|
| `images` | JPEG data URLs for `traditional`, `learned`, `hybrid` and `reference` |
| `metrics` | PSNR and CIE76 ΔE per arm. `null` for uploads, which have no reference |
| `route`, `model` | The model that was used. `fast` is the compact hybrid. On `quality`, the hybrid image is the learned model's output |
| `stats` | Measured noise and brightness |
| `controls` | Adaptive settings and the routing reason |
| `gains` | Applied white-balance gains |
| `timings_ms`, `total_ms` | Timing |
| `simulated` | `true` for uploads |
| `logged` | Whether the run reached MongoDB |

The live API reports CIE76 ΔE, because scikit-image is not deployed. Published results use CIEDE2000, so the two are not directly comparable.

| Status | Meaning |
|---|---|
| 400 | Missing input, invalid sample ID, or an undecodable image |
| 404 | Unknown sample |
| 422 | Invalid field value |

## `GET /api/results`

Returns the latest evaluation summary from MongoDB, or from `results/summary.json`. Returns 404 until results exist.

## `GET /api/telemetry`

Returns aggregate usage.

```json
{ "database": "connected", "total": 12, "by_route": [], "recent": [] }
```
