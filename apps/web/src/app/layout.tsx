import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "VietTopic AI — Phân loại chủ đề tiếng Việt",
  description: "Trải nghiệm mô hình AI phân loại văn bản tiếng Việt thành 25 lĩnh vực với kết quả trực tiếp từ FastAPI.",
};

const themeBootstrap = '(function(){try{var saved=localStorage.getItem("viettopic-theme");var system=window.matchMedia("(prefers-color-scheme: dark)").matches?"dark":"light";document.documentElement.dataset.theme=saved==="light"||saved==="dark"?saved:system;}catch(e){document.documentElement.dataset.theme="light";}})();';

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="vi" suppressHydrationWarning>
      <head><script dangerouslySetInnerHTML={{ __html: themeBootstrap }} /></head>
      <body>{children}</body>
    </html>
  );
}
