import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = {
  title: "LumaForge",
  description: "A hybrid learned smartphone ISP: classical camera stages plus a compact neural restorer.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <nav className="nav">
          <div className="wrap nav-inner">
            <Link href="/" className="brand">
              Luma<span>Forge</span>
            </Link>
            <div className="nav-links">
              <Link href="/">Lab</Link>
              <Link href="/results">Results</Link>
              <Link href="/architecture">Architecture</Link>
              <a href="https://github.com/Charis-E-Shiny/LumaForge">GitHub</a>
            </div>
          </div>
        </nav>
        <main className="wrap">{children}</main>
        <footer className="footer">
          <div className="wrap">Team Asphalt (T32) · Problem P2: Intelligent and Computationally Efficient Smartphone ISP</div>
        </footer>
      </body>
    </html>
  );
}
