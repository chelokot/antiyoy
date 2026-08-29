import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({ variable: "--font-geist-sans", subsets: ["latin"] });
const geistMono = Geist_Mono({ variable: "--font-geist-mono", subsets: ["latin"] });

export const metadata: Metadata = {
  metadataBase: new URL("https://antiyoy-arena-lab.chelokot.chatgpt.site"),
  title: "Antiyoy Arena Lab",
  description: "Inspect deterministic self-play, policies, state, and ratings from the Antiyoy RL environment.",
  openGraph: {
    title: "Antiyoy Arena Lab",
    description: "Deterministic Rust self-play, compiled to WebAssembly.",
    images: [{ url: "/og.png", width: 1672, height: 941 }],
  },
  twitter: {
    card: "summary_large_image",
    title: "Antiyoy Arena Lab",
    description: "Deterministic Rust self-play, compiled to WebAssembly.",
    images: ["/og.png"],
  },
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body className={`${geistSans.variable} ${geistMono.variable} antialiased`}>{children}</body></html>;
}
