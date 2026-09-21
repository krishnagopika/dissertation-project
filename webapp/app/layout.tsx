import type { Metadata } from "next";
import { SessionProvider } from "next-auth/react";
import { ThemeProvider } from "next-themes";

import { TooltipProvider } from "@/components/ui/tooltip";
import "./globals.css";

export const metadata: Metadata = {
  title: "Speech Emotion Pipeline",
  description:
    "Multimodal emotion and sentiment recognition from speech: Voxtral " +
    "acoustic features fused with XLM-RoBERTa over ASR transcripts, with " +
    "VAD and WER diagnostics.",
};

/**
 * Root layout.
 *
 * `suppressHydrationWarning` on <html> is required by next-themes: it writes
 * the theme class before React hydrates, so the server and client markup
 * differ by that one attribute on purpose.
 *
 * @param children - The page being rendered.
 * @returns The HTML shell.
 */
export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className="min-h-screen antialiased">
        <ThemeProvider
          attribute="class"
          defaultTheme="system"
          enableSystem
          disableTransitionOnChange
        >
          <SessionProvider>
            <TooltipProvider delayDuration={200}>{children}</TooltipProvider>
          </SessionProvider>
        </ThemeProvider>
      </body>
    </html>
  );
}
