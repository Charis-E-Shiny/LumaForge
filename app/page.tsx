import Lab from "@/components/Lab";
import Telemetry from "@/components/Telemetry";

export default function Home() {
  return (
    <>
      <section className="hero">
        <h1>RAW in. Faithful colour out.</h1>
        <p>
          LumaForge renders noisy smartphone sensor data with a classical ISP, then lets a compact neural network correct
          what the classical stages cannot. Choose a real test capture, degrade it in the RAW domain, and compare every arm
          of the pipeline.
        </p>
      </section>
      <Lab />
      <Telemetry />
    </>
  );
}
