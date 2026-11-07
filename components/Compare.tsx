"use client";

import { useRef, useState } from "react";

type Props = { before: string; after: string; beforeLabel: string; afterLabel: string };

export default function Compare({ before, after, beforeLabel, afterLabel }: Props) {
  const [position, setPosition] = useState(50);
  const frame = useRef<HTMLDivElement>(null);

  function move(clientX: number) {
    const box = frame.current?.getBoundingClientRect();
    if (box) setPosition(Math.min(100, Math.max(0, ((clientX - box.left) / box.width) * 100)));
  }

  return (
    <div
      ref={frame}
      className="compare"
      role="slider"
      tabIndex={0}
      aria-label={`Compare ${beforeLabel} with ${afterLabel}`}
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={Math.round(position)}
      onPointerDown={(event) => {
        event.currentTarget.setPointerCapture(event.pointerId);
        move(event.clientX);
      }}
      onPointerMove={(event) => {
        if (event.buttons) move(event.clientX);
      }}
      onKeyDown={(event) => {
        if (event.key === "ArrowLeft") setPosition((p) => Math.max(0, p - 2));
        if (event.key === "ArrowRight") setPosition((p) => Math.min(100, p + 2));
      }}
    >
      <img src={after} alt={afterLabel} />
      <img src={before} alt={beforeLabel} className="top" style={{ clipPath: `inset(0 ${100 - position}% 0 0)` }} />
      <div className="divider" style={{ left: `${position}%` }} />
      <span className="tag left">{beforeLabel}</span>
      <span className="tag right">{afterLabel}</span>
    </div>
  );
}
