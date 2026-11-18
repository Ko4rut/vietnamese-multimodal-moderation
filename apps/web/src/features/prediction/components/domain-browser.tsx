"use client";

import { useMemo, useState } from "react";
import { Icon } from "@/components/ui/icon";
import { displayDomain, REFERENCE_DOMAINS } from "../data/domains";
import type { DomainLabel } from "../types";

interface DomainBrowserProps {
  domains: DomainLabel[];
}

export function DomainBrowser({ domains }: DomainBrowserProps) {
  const [search, setSearch] = useState("");
  const apiConnected = domains.length > 0;
  const source = apiConnected ? domains : REFERENCE_DOMAINS;
  const filtered = useMemo(() => source.filter((item) => {
    const query = search.trim().toLocaleLowerCase("vi");
    return (displayDomain(item.name) + " " + item.name).toLocaleLowerCase("vi").includes(query);
  }), [source, search]);

  return (
    <section className="domains-section" id="domains" aria-labelledby="domains-heading">
      <div className="section-bar">
        <div><span className="section-eyebrow">DANH MỤC / 02</span><h2 id="domains-heading">Khám phá 25 lĩnh vực <span className="heading-highlight">AI nhận diện</span></h2>
          <p>Các nhãn phân loại thuộc bộ dữ liệu VTSNLP.</p></div>
        <span className="domain-source"><span className="status-dot"/>{apiConnected ? "Đồng bộ với API" : "Danh mục tham khảo"}</span>
      </div>
      <div className="domain-panel">
        <div className="domain-panel-head">
          <div className="domain-count"><Icon name="grid" size={19}/><strong>{filtered.length}</strong><span>/ {source.length} lĩnh vực</span></div>
          <label className="domain-search"><span className="sr-only">Tìm lĩnh vực</span>
            <input value={search} onChange={(event) => setSearch(event.target.value)} placeholder="Tìm kiếm lĩnh vực..." type="search"/>
          </label>
        </div>
        <div className="domain-grid">
          {filtered.map((domain) => (
            <div className="domain-item" key={domain.id}>
              <span className="domain-number">{String(domain.id + 1).padStart(2, "0")}</span>
              <div className="domain-label"><strong>{displayDomain(domain.name)}</strong><small>{domain.name.replaceAll("_", " ")}</small></div>
              <Icon name="chevron-right" size={16}/>
            </div>
          ))}
          {filtered.length === 0 && <p className="no-domains">Không tìm thấy lĩnh vực phù hợp.</p>}
        </div>
      </div>
    </section>
  );
}
