export const metadata = { title: "Architecture · LumaForge" };

const STAGES = [
  ["1. Scene analysis", "Fits a shot and read noise model from the two green planes, and measures brightness and clipping."],
  ["2. Raw denoise", "Variance-stabilised guided filter. Strength follows the measured noise."],
  ["3. White balance", "Shades-of-grey estimate shrunk towards a per-camera prior."],
  ["4. Demosaic", "Malvar-He-Cutler gradient-corrected interpolation."],
  ["5. Colour and tone", "Fitted camera-to-sRGB CCM, tone curve on luminance, sRGB gamma."],
  ["6. Chroma filter", "Edge-preserving chroma smoothing in YCbCr, guided by luma."],
  ["7. Neural refinement", "LumaNet predicts a global colour correction and a local residual on top of the render."],
  ["8. Adaptive routing", "Low post-gain noise uses the compact int8 model. High noise uses the larger fp16 model."],
];

export default function ArchitecturePage() {
  return (
    <>
      <section className="hero">
        <h1>How the hybrid pipeline works</h1>
        <p>
          The classical stages provide a physically meaningful render. The network only learns what they get wrong, which
          keeps it small enough for a phone and anchors colour to the sensor data.
        </p>
      </section>
      <section className="section">
        <img className="figure" src="/figures/architecture.svg" alt="System diagram from Bayer RAW through classical stages and LumaNet to sRGB output" />
      </section>
      <section className="section">
        <h2>Processing order</h2>
        <div className="flow">
          {STAGES.map(([title, text]) => (
            <div className="panel" key={title}>
              <h3>{title}</h3>
              <p className="note" style={{ marginTop: 6 }}>{text}</p>
            </div>
          ))}
        </div>
      </section>
      <section className="section">
        <h2>Deployment</h2>
        <p>
          The Next.js interface and a Python function run on Vercel. Inference uses ONNX Runtime with NumPy, without
          PyTorch, so the function stays small. MongoDB Atlas stores run telemetry and evaluation results. If the database
          is unavailable, processing still works.
        </p>
      </section>
    </>
  );
}
