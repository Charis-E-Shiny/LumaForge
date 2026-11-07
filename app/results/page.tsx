import Results from "@/components/Results";

export const metadata = { title: "Results · LumaForge" };

export default function ResultsPage() {
  return (
    <>
      <section className="hero">
        <h1>Evaluation results</h1>
        <p>
          Every figure comes from the held-out SIDD-CC test scenes, which were never used for training, calibration or
          tuning. White balance is estimated blindly because the test split carries no capture metadata.
        </p>
      </section>
      <Results />
    </>
  );
}
