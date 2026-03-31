export const metadata = {
  title: "TakNN - Play Tak",
  description: "Browser-based Tak game",
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
