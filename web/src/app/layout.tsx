import type { Metadata, Viewport } from "next";

export const metadata: Metadata = {
  title: "TakNN - Play Tak",
  description: "Browser-based Tak game with neural network AI",
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body style={{ margin: 0, fontFamily: "system-ui, sans-serif", backgroundColor: "#fafafa" }}>
        {children}
      </body>
    </html>
  );
}
