"use client";

import { Icon } from "@/components/ui/icon";

interface SidebarProps {
  open: boolean;
  onClose: () => void;
}

const navItems = [
  { href: "#playground", icon: "sparkles" as const, label: "AI Playground" },
  { href: "#model", icon: "chart" as const, label: "Thông tin mô hình" },
  { href: "#domains", icon: "grid" as const, label: "25 lĩnh vực" },
];

export function Sidebar({ open, onClose }: SidebarProps) {
  return (
    <>
      {open && <button className="sidebar-overlay" aria-label="Đóng menu" onClick={onClose} type="button" />}
      <aside className={"sidebar" + (open ? " sidebar-open" : "")}>
        <div className="sidebar-inner">
          <div className="brand-row">
            <a href="#playground" className="brand" onClick={onClose}>
              <span className="brand-mark"><Icon name="sparkles" size={23} /></span>
              <span><strong>viettopic<span className="brand-accent">.ai</span></strong><small>INTELLIGENCE STUDIO</small></span>
            </a>
            <button className="sidebar-close icon-button" type="button" onClick={onClose} aria-label="Đóng menu">
              <Icon name="close" size={20} />
            </button>
          </div>

          <div className="workspace-label">WORKSPACE</div>
          <nav className="side-nav" aria-label="Điều hướng chính">
            {navItems.map((item, index) => (
              <a key={item.href} href={item.href} onClick={onClose} className={"nav-item" + (index === 0 ? " nav-active" : "")}>
                <Icon name={item.icon} size={19} /><span>{item.label}</span>
                {index === 0 && <span className="nav-active-dot" />}
              </a>
            ))}
          </nav>

          <div className="sidebar-spacer" />
          <div className="sidebar-callout">
            <div className="sidebar-callout-icon"><Icon name="brain" size={22} /></div>
            <p className="callout-title">Khám phá sức mạnh NLP</p>
            <p>Phân tích ngữ nghĩa văn bản tiếng Việt qua mô hình machine learning.</p>
            <a href="#playground" onClick={onClose}>Thử ngay <Icon name="arrow-right" size={15} /></a>
          </div>

          <a className="sidebar-repo" href="https://github.com/Ko4rut/vietnamese-multimodal-moderation"
            target="_blank" rel="noopener noreferrer">
            <Icon name="github" size={18}/><span>GitHub Repository</span><Icon name="external" size={14}/>
          </a>
          <div className="sidebar-footer">VietTopic AI <span>•</span> Research demo</div>
        </div>
      </aside>
    </>
  );
}
