"use client";

import { useEffect, useState } from "react";

type Row = { route: string; runs: number; avg_total_ms: number; avg_hybrid_ms: number; avg_psnr: number | null };
type Data = { database: string; total: number; by_route: Row[] };

const STATUS: Record<string, string> = {
  connected: "MongoDB connected",
  not_configured: "MongoDB not configured",
  unreachable: "MongoDB unreachable",
};

export default function Telemetry() {
  const [data, setData] = useState<Data | null>(null);

  useEffect(() => {
    fetch("/api/telemetry")
      .then((response) => (response.ok ? response.json() : null))
      .then(setData)
      .catch(() => setData(null));
  }, []);

  if (!data) return null;
  const connected = data.database === "connected";

  return (
    <section className="section">
      <div style={{ display: "flex", gap: 10, alignItems: "center", flexWrap: "wrap" }}>
        <h2>Live usage</h2>
        <span className={`pill ${connected ? "on" : "off"}`}>{STATUS[data.database] ?? data.database}</span>
      </div>
      {data.by_route.length > 0 ? (
        <div className="panel table-wrap">
          <table>
            <thead>
              <tr><th>Route</th><th>Runs</th><th>Avg total (ms)</th><th>Avg inference (ms)</th><th>Avg PSNR (dB)</th></tr>
            </thead>
            <tbody>
              {data.by_route.map((row) => (
                <tr key={row.route}>
                  <td>{row.route}</td>
                  <td>{row.runs}</td>
                  <td>{row.avg_total_ms.toFixed(0)}</td>
                  <td>{row.avg_hybrid_ms.toFixed(0)}</td>
                  <td>{row.avg_psnr == null ? "–" : row.avg_psnr.toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <p className="note">Run metadata appears here once the pipeline has been used. Images are never stored.</p>
      )}
    </section>
  );
}
