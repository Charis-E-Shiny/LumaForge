/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // In production vercel.json routes /api to the Python function.
  async rewrites() {
    if (process.env.NODE_ENV !== "development") return [];
    return [{ source: "/api/:path*", destination: "http://127.0.0.1:8000/api/:path*" }];
  },
};

export default nextConfig;
