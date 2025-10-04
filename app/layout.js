//layout.js: root layout component that wraps all pages with header and global styles
import "./globals.css";

//main layout component that provides the app structure and header
export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-background font-sans antialiased">
        <div className="flex min-h-screen flex-col">
          {/*app header with logo and branding*/}
          <header className="scout-header px-4 py-4">
            <div className="mx-auto max-w-7xl">
              <div className="flex items-center justify-between">
                <img 
                  src="/logo.png" 
                  alt="SCOUT AI Logo" 
                  className="h-12 w-auto -ml-24"
                />
                <div className="text-center">
                  <h1 className="text-2xl font-bold text-primary-foreground scout-wings tracking-wider">
                    Scout AI
                  </h1>
                  <p className="text-primary-foreground/80 text-sm mt-1 font-medium">
                    AI Content Detection
                  </p>
                </div>
                <div className="w-12"></div> {/*spacer for layout balance*/}
              </div>
            </div>
          </header>
          {/*main content area where pages are rendered*/}
          <main className="flex-1 bg-background">{children}</main>
        </div>
      </body>
    </html>
  );
}
