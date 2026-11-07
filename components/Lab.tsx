"use client";

import { useEffect, useState } from "react";
import Compare from "./Compare";

type Sample = { id: string; title: string; camera: string; iso: string; thumbnail: string };
type Arm = "traditional" | "learned" | "hybrid" | "reference";
type Result = {
  images: Record<Arm, string | null>;
  metrics: Record<string, { psnr: number; delta_e: number }> | null;
  route: string;
  model: string;
  stats: { sigma: number; p99: number; clipped: number };
  controls: { exposure: number; sharpen: number; reason: string };
  gains: number[];
  timings_ms: Record<string, number>;
  total_ms: number;
  simulated: boolean;
  logged: boolean;
};

const LABELS: Record<Arm, string> = {
  traditional: "Traditional ISP",
  learned: "Learned ISP",
  hybrid: "Hybrid ISP",
  reference: "Ground truth",
};

const MAX_UPLOAD_SIDE = 768;

/** Shrinks an upload in the browser, which keeps it under the function's 4.5 MB body limit. */
async function shrink(file: File): Promise<string> {
  const bitmap = await createImageBitmap(file);
  const scale = Math.min(1, MAX_UPLOAD_SIDE / Math.max(bitmap.width, bitmap.height));
  const canvas = document.createElement("canvas");
  canvas.width = Math.round(bitmap.width * scale);
  canvas.height = Math.round(bitmap.height * scale);
  canvas.getContext("2d")?.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
  return canvas.toDataURL("image/jpeg", 0.92);
}

export default function Lab() {
  const [samples, setSamples] = useState<Sample[]>([]);
  const [sampleId, setSampleId] = useState<string | null>(null);
  const [upload, setUpload] = useState<string | null>(null);
  const [stops, setStops] = useState(0);
  const [noise, setNoise] = useState(0);
  const [cast, setCast] = useState(0);
  const [mode, setMode] = useState("auto");
  const [left, setLeft] = useState<Arm>("traditional");
  const [result, setResult] = useState<Result | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    fetch("/api/samples")
      .then((response) => response.json())
      .then((rows: Sample[]) => {
        setSamples(rows);
        if (rows.length) setSampleId(rows[0].id);
      })
      .catch(() => setError("The processing API is not reachable right now."));
  }, []);

  async function run() {
    setBusy(true);
    setError("");
    try {
      const response = await fetch("/api/process", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          sample_id: upload ? null : sampleId,
          image_base64: upload,
          mode,
          degradation: { exposure_stops: stops, noise, cast },
        }),
      });
      if (!response.ok) throw new Error(`Request failed with status ${response.status}`);
      const payload: Result = await response.json();
      setResult(payload);
      if (payload.simulated && left === "reference") setLeft("traditional");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Processing failed.");
    } finally {
      setBusy(false);
    }
  }

  async function onFile(file?: File) {
    if (!file) return;
    setError("");
    try {
      setUpload(await shrink(file));
      setSampleId(null);
    } catch {
      setError("That file could not be read as an image.");
    }
  }

  const arms: Arm[] = result?.images.reference ? ["traditional", "learned", "reference"] : ["traditional", "learned"];

  return (
    <section className="lab" aria-label="RAW processing lab">
      <div className="panel stack">
        <div className="stack" style={{ gap: 8 }}>
          <span className="label">Test capture</span>
          <div className="samples">
            {samples.map((sample) => (
              <button
                key={sample.id}
                className="sample"
                aria-pressed={sample.id === sampleId}
                title={`${sample.title} · ${sample.camera} · ISO ${sample.iso}`}
                onClick={() => {
                  setSampleId(sample.id);
                  setUpload(null);
                }}
              >
                <img src={sample.thumbnail} alt={sample.title} />
              </button>
            ))}
          </div>
          <label className="upload">
            {upload ? "Own image selected" : "Or use your own image"}
            <input type="file" accept="image/*" onChange={(event) => onFile(event.target.files?.[0])} />
          </label>
          {upload && <p className="note">Uploads are converted to a simulated mosaic, not true RAW, and are never stored.</p>}
        </div>

        <div className="stack" style={{ gap: 10 }}>
          <span className="label">Sensor conditions</span>
          <label className="field">
            <span className="field-row">Underexposure <b>{stops.toFixed(1)} stops</b></span>
            <input type="range" min={0} max={4} step={0.5} value={stops} onChange={(e) => setStops(+e.target.value)} />
          </label>
          <label className="field">
            <span className="field-row">Added sensor noise <b>{Math.round(noise * 100)}%</b></span>
            <input type="range" min={0} max={1} step={0.05} value={noise} onChange={(e) => setNoise(+e.target.value)} />
          </label>
          <label className="field">
            <span className="field-row">Colour cast <b>{cast === 0 ? "none" : cast > 0 ? "warm" : "cool"}</b></span>
            <input type="range" min={-1} max={1} step={0.1} value={cast} onChange={(e) => setCast(+e.target.value)} />
          </label>
        </div>

        <div className="stack" style={{ gap: 8 }}>
          <span className="label">Model route</span>
          <div className="segmented">
            {["auto", "fast", "quality"].map((value) => (
              <button key={value} aria-pressed={mode === value} onClick={() => setMode(value)}>
                {value[0].toUpperCase() + value.slice(1)}
              </button>
            ))}
          </div>
        </div>

        <button className="button" onClick={run} disabled={busy || (!sampleId && !upload)}>
          {busy ? "Processing…" : "Run pipeline"}
        </button>
        {error && <p className="error" role="alert">{error}</p>}
      </div>

      <div className="stack">
        <div className="panel">
          {result?.images.hybrid ? (
            <div className="stack" style={{ gap: 10 }}>
              <div className="segmented" aria-label="Comparison image">
                {arms.map((arm) => (
                  <button key={arm} aria-pressed={left === arm} onClick={() => setLeft(arm)}>
                    {LABELS[arm]}
                  </button>
                ))}
              </div>
              <Compare
                before={result.images[left] ?? result.images.traditional!}
                after={result.images.hybrid}
                beforeLabel={LABELS[left]}
                afterLabel={LABELS.hybrid}
              />
            </div>
          ) : (
            <div className="empty">{busy ? "Rendering all three pipelines…" : "Run the pipeline to compare the outputs."}</div>
          )}
        </div>

        {result && (
          <div className="panel stack">
            {result.metrics && (
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr><th>Against ground truth</th><th>PSNR (dB)</th><th>ΔE (CIE76)</th></tr>
                  </thead>
                  <tbody>
                    {Object.entries(result.metrics).map(([name, row]) => (
                      <tr key={name} className={name === "hybrid" ? "highlight" : ""}>
                        <td>{LABELS[name as Arm]}</td>
                        <td>{row.psnr.toFixed(2)}</td>
                        <td>{row.delta_e.toFixed(2)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
            <div className="tiles">
              <div className="tile"><b>{result.route}</b><span>adaptive route</span></div>
              <div className="tile"><b>{result.stats.sigma.toFixed(4)}</b><span>measured noise σ</span></div>
              <div className="tile"><b>{result.controls.exposure.toFixed(1)}×</b><span>exposure gain</span></div>
              <div className="tile"><b>{result.timings_ms.hybrid_net.toFixed(0)} ms</b><span>hybrid inference</span></div>
              <div className="tile"><b>{result.timings_ms.classical.toFixed(0)} ms</b><span>classical stages</span></div>
            </div>
            <p className="note">
              {result.controls.reason}. White-balance gains R {result.gains[0].toFixed(2)}, B {result.gains[2].toFixed(2)}.
              {result.simulated && " Input was simulated from an RGB image."}
            </p>
          </div>
        )}
      </div>
    </section>
  );
}
