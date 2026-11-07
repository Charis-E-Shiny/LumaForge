"use client";

import { useEffect, useState } from "react";

type Metrics = { psnr: number; ssim: number; delta_e: number; lpips: number };
type Row = Metrics & { id: string; label: string };
type Latency = { ms_median: number; peak_mb: number };
type Model = { params: number; size_mb: number; gmacs_512: number; latency: Record<string, Latency> };
type Summary = {
  test: { blocks: number; scenes: number };
  routed: { fast: number; quality: number };
  main: Row[];
  robust: { id: string; label: string; arms: Record<string, Metrics> }[];
  ablation: { classical: Row[]; trained_clean: Row[]; trained_severe: Row[] };
  benchmark: { machine: { cpu: string }; classical_ms: Record<string, number>; models: Record<string, Model> };
};

function MetricTable({ rows, highlight }: { rows: Row[]; highlight: string }) {
  return (
    <div className="panel table-wrap">
      <table>
        <thead>
          <tr><th>Configuration</th><th>PSNR (dB)</th><th>SSIM</th><th>ΔE2000</th><th>LPIPS</th></tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.id} className={row.id === highlight ? "highlight" : ""}>
              <td>{row.label}</td>
              <td>{row.psnr.toFixed(2)}</td>
              <td>{row.ssim.toFixed(4)}</td>
              <td>{row.delta_e.toFixed(2)}</td>
              <td>{row.lpips.toFixed(4)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function Results() {
  const [data, setData] = useState<Summary | null>(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    fetch("/api/results")
      .then((response) => {
        if (!response.ok) throw new Error();
        return response.json();
      })
      .then(setData)
      .catch(() => setFailed(true));
  }, []);

  if (failed) return <p className="error">Results could not be loaded.</p>;
  if (!data) return <p className="note">Loading results…</p>;

  const models = Object.entries(data.benchmark.models);
  const labels = Object.fromEntries(data.main.map((row) => [row.id, row.label]));

  return (
    <>
      <section className="section">
        <h2>Traditional, learned and hybrid</h2>
        <p>{data.test.blocks} blocks from {data.test.scenes} held-out scenes. Higher PSNR and SSIM are better; lower ΔE and LPIPS are better.</p>
        <MetricTable rows={data.main} highlight="hybrid_adaptive" />
        <img className="figure" src="/figures/comparison.png" alt="Side-by-side crops from the traditional, learned and hybrid pipelines against ground truth" />
      </section>

      <section className="section">
        <h2>Robustness</h2>
        <p>Degradations are applied in the RAW domain with fixed seeds, so every pipeline receives identical input.</p>
        <img className="figure" src="/figures/robustness.png" alt="PSNR of each pipeline across degraded conditions" />
        <div className="panel table-wrap">
          <table>
            <thead>
              <tr><th>Condition</th><th>Traditional PSNR</th><th>Learned PSNR</th><th>Hybrid PSNR</th><th>Traditional ΔE</th><th>Learned ΔE</th><th>Hybrid ΔE</th></tr>
            </thead>
            <tbody>
              {data.robust.map((row) => (
                <tr key={row.id}>
                  <td>{row.label}</td>
                  <td>{row.arms.traditional.psnr.toFixed(2)}</td>
                  <td>{row.arms.learned?.psnr.toFixed(2) ?? "–"}</td>
                  <td>{row.arms.hybrid_adaptive.psnr.toFixed(2)}</td>
                  <td>{row.arms.traditional.delta_e.toFixed(2)}</td>
                  <td>{row.arms.learned?.delta_e.toFixed(2) ?? "–"}</td>
                  <td>{row.arms.hybrid_adaptive.delta_e.toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="section">
        <h2>Computational cost</h2>
        <p>
          Measured on {data.benchmark.machine.cpu}, CPU only, with ONNX Runtime. Classical stages add{" "}
          {data.benchmark.classical_ms["512"].toFixed(0)} ms at 512 px.
        </p>
        <div className="panel table-wrap">
          <table>
            <thead>
              <tr><th>Model</th><th>Parameters</th><th>GMACs (512 px)</th><th>Size (MB)</th><th>512 px, 1 thread (ms)</th><th>512 px, 4 threads (ms)</th><th>Peak memory (MB)</th></tr>
            </thead>
            <tbody>
              {models.map(([name, row]) => (
                <tr key={name}>
                  <td>{labels[name] ?? name}</td>
                  <td>{row.params.toLocaleString("en-US")}</td>
                  <td>{row.gmacs_512.toFixed(2)}</td>
                  <td>{row.size_mb.toFixed(2)}</td>
                  <td>{row.latency["512px_1t"].ms_median.toFixed(0)}</td>
                  <td>{row.latency["512px_4t"].ms_median.toFixed(0)}</td>
                  <td>{row.latency["512px_1t"].peak_mb.toFixed(0)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="section">
        <h2>Ablation study</h2>
        <h3>Classical stages removed</h3>
        <MetricTable rows={data.ablation.classical} highlight="full" />
        <h3>Trained variants, clean captures</h3>
        <MetricTable rows={data.ablation.trained_clean} highlight="abl_base" />
        <h3>Trained variants, combined severe condition</h3>
        <MetricTable rows={data.ablation.trained_severe} highlight="abl_base" />
      </section>
    </>
  );
}
