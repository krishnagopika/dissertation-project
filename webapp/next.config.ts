import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Nothing exotic: this app is a thin authenticated proxy in front of Modal,
  // so the defaults are correct. Kept as a file rather than omitted so the
  // serverRuntimeConfig story is explicit for whoever deploys it next.
  reactStrictMode: true,
};

export default nextConfig;
