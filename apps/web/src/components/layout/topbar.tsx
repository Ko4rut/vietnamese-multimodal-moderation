import { ThemeToggle } from "@/components/theme/theme-toggle";
import { Icon } from "@/components/ui/icon";

interface TopbarProps {
  onOpenMenu: () => void;
  ready: boolean | null;
}

export function Topbar({ onOpenMenu, ready }: TopbarProps) {
  return (
    <header className="topbar">
      <div className="topbar-left">
        <button className="icon-button mobile-menu" type="button" onClick={onOpenMenu} aria-label="Mở menu">
          <Icon name="menu" />
        </button>
        <div className="breadcrumb"><span>Workspace</span><Icon name="chevron-right" size={15}/><strong>AI Playground</strong></div>
      </div>
      <div className="topbar-actions">
        <div className={"connection-pill " + (ready === true ? "connection-online" : ready === false ? "connection-offline" : "connection-loading")}>
          <span className="status-dot" />
          <span>{ready === true ? "Model online" : ready === false ? "Model offline" : "Đang kết nối"}</span>
        </div>
        <span className="topbar-divider" />
        <ThemeToggle />
        <span className="avatar" aria-label="VietTopic AI"><Icon name="sparkles" size={18}/></span>
      </div>
    </header>
  );
}
