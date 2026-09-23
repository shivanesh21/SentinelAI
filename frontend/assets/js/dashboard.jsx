/* SentinelAI React Dashboard — dashboard.jsx
   All panels connected to FastAPI backend (/api/*).
   No build step; runs directly in browser via Babel. */

const { useState, useEffect, useCallback, useRef } = React;

// ─── Config ──────────────────────────────────────────────────
const API = "http://localhost:8000/api";
const REFRESH_MS = 20000;

// ─── Utilities ───────────────────────────────────────────────
function pct(v) {
  if (v == null) return "—";
  return (v * 100).toFixed(1) + "%";
}
function f2(v) {
  if (v == null) return "—";
  return v.toFixed ? v.toFixed(2) : v;
}
function f4(v) {
  if (v == null) return "—";
  return v.toFixed ? v.toFixed(4) : v;
}
function num(v) {
  if (v == null) return "—";
  return v.toLocaleString();
}
function relTime(iso) {
  if (!iso) return "—";
  const diff = Date.now() - new Date(iso).getTime();
  const s = Math.floor(diff / 1000);
  if (s < 60) return `${s}s ago`;
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
  return `${Math.floor(s / 86400)}d ago`;
}
function shortTs(iso) {
  if (!iso) return "—";
  return new Date(iso).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

async function apiFetch(path, opts = {}) {
  const res = await fetch(API + path, { cache: "no-store", ...opts });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text);
  }
  return res.json();
}

// ─── Toast system ────────────────────────────────────────────
function ToastContainer({ toasts }) {
  return (
    <div className="toast-container">
      {toasts.map(t => (
        <div key={t.id} className={`toast ${t.type}`}>{t.msg}</div>
      ))}
    </div>
  );
}

function useToasts() {
  const [toasts, setToasts] = useState([]);
  const add = useCallback((msg, type = "info") => {
    const id = Date.now() + Math.random();
    setToasts(prev => [...prev, { id, msg, type }]);
    setTimeout(() => setToasts(prev => prev.filter(t => t.id !== id)), 3500);
  }, []);
  return { toasts, add };
}

// ─── Score Bar ───────────────────────────────────────────────
function ScoreBar({ value, max = 1, color }) {
  const w = Math.min(100, Math.max(0, (value / max) * 100));
  const bg = color || (w > 70 ? "var(--bad)" : w > 40 ? "var(--warn)" : "var(--ok)");
  return (
    <div className="score-bar-wrap">
      <div className="score-bar-track">
        <div className="score-bar-fill" style={{ width: `${w}%`, background: bg }} />
      </div>
      <span className="score-val">{f2(value)}</span>
    </div>
  );
}

// ─── Risk Badge ──────────────────────────────────────────────
function RiskBadge({ level }) {
  const l = (level || "unknown").toLowerCase();
  const icons = { normal: "✓", warning: "⚠", anomalous: "◉", critical: "⛔", unknown: "?" };
  return (
    <span className={`risk-badge ${l}`}>{icons[l] || "?"} {level || "UNKNOWN"}</span>
  );
}

// ─── Drift Badge ─────────────────────────────────────────────
function DriftBadge({ level }) {
  const l = (level || "none").toLowerCase();
  const icons = { none: "●", moderate: "◆", significant: "▲" };
  return (
    <span className={`drift-badge ${l}`}>{icons[l] || "●"} {level}</span>
  );
}

// ─── KPI Card ────────────────────────────────────────────────
function KpiCard({ icon, label, value, sub, variant = "" }) {
  return (
    <div className={`kpi-card ${variant}`}>
      <div className="kpi-icon">{icon}</div>
      <div className="kpi-label">{label}</div>
      <div className="kpi-value">{value}</div>
      {sub && <div className="kpi-sub">{sub}</div>}
    </div>
  );
}

// ─── Panel Wrapper ───────────────────────────────────────────
function Panel({ title, icon, badge, actions, children }) {
  return (
    <div className="panel">
      <div className="panel-header">
        <span className="panel-title">{icon && <span>{icon}</span>}{title}</span>
        <div className="flex-center gap-8">
          {badge}
          {actions}
        </div>
      </div>
      <div className="panel-body">{children}</div>
    </div>
  );
}

// ─── Empty State ─────────────────────────────────────────────
function Empty({ icon = "🔍", msg = "No data available" }) {
  return (
    <div className="empty">
      <div className="empty-icon">{icon}</div>
      <div>{msg}</div>
    </div>
  );
}

// ─── Skeleton ────────────────────────────────────────────────
function Skel({ h = 20, w = "100%", mb = 8 }) {
  return <div className="skeleton" style={{ height: h, width: w, marginBottom: mb }} />;
}

// ═══════════════════════════════════════════════════════════════
// OVERVIEW TAB
// ═══════════════════════════════════════════════════════════════
function OverviewTab({ data, drift }) {
  const { health, overview, risk } = data;

  const flagged = risk?.n_flagged ?? 0;
  const services = risk?.services ?? [];
  const driftStatus = drift?.trigger?.retraining_required;
  const driftSeverity = drift?.trigger?.severity;

  return (
    <div className="page-enter">
      {/* KPI Bar */}
      <div className="kpi-grid">
        <KpiCard
          icon="⚡"
          label="API Status"
          value={health?.status === "ok" ? "UP" : "DOWN"}
          sub={health ? "sentinelai-backend" : "unreachable"}
          variant={health?.status === "ok" ? "ok" : "bad"}
        />
        <KpiCard
          icon="🗄"
          label="Feature Rows"
          value={num(overview?.n_rows)}
          sub={`${overview?.n_features || "—"} features × ${overview?.n_labels || "—"} labels`}
          variant="info"
        />
        <KpiCard
          icon="🛰"
          label="Services Tracked"
          value={num(overview?.n_services)}
          sub={`${flagged} at risk`}
          variant={flagged > 0 ? "warn" : "ok"}
        />
        <KpiCard
          icon="🎯"
          label="Prediction Window"
          value={overview?.prediction_window_min ? `${overview.prediction_window_min}m` : "—"}
          sub="failure horizon"
          variant=""
        />
        <KpiCard
          icon="🔬"
          label="Data Drift"
          value={driftStatus ? "⚠ Detected" : drift ? "Stable" : "—"}
          sub={drift ? `${drift.data_drift?.drifted_features_count}/${drift.data_drift?.total_features} features drifted` : "no report yet"}
          variant={driftStatus ? (driftSeverity === "critical" ? "bad" : "warn") : "ok"}
        />
      </div>

      <div className="two-col">
        {/* Service Health */}
        <Panel title="Service Risk Overview" icon="🛡" badge={
          <span className={`risk-badge ${flagged > 0 ? "warning" : "normal"}`}>
            {flagged} flagged
          </span>
        }>
          {services.length === 0 ? (
            <Empty icon="🔍" msg="No service data. Start the risk engine." />
          ) : (
            <table className="service-table">
              <thead>
                <tr>
                  <th>Service</th>
                  <th>Risk Score</th>
                  <th>Anomaly</th>
                  <th>Level</th>
                </tr>
              </thead>
              <tbody>
                {services.map(svc => (
                  <tr key={svc.service}>
                    <td>
                      <span className={`status-pulse ${svc.risk ? "bad" : "ok"}`} />
                      <span style={{ fontWeight: 600 }}>{svc.service}</span>
                    </td>
                    <td><ScoreBar value={svc.risk_score} /></td>
                    <td>
                      {svc.anomaly_score != null
                        ? <ScoreBar value={svc.anomaly_score} color="var(--crit)" />
                        : <span className="text-muted">—</span>
                      }
                    </td>
                    <td><RiskBadge level={svc.risk_level} /></td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </Panel>

        {/* Dataset Splits */}
        <Panel title="Dataset Splits" icon="📊">
          {!overview?.splits ? (
            <Empty msg="No split report available." />
          ) : (
            <>
              <table className="service-table" style={{ marginBottom: 16 }}>
                <thead>
                  <tr>
                    <th>Split</th>
                    <th>Rows</th>
                    <th>Positive Rate</th>
                  </tr>
                </thead>
                <tbody>
                  {["train", "val", "test"].map(name => {
                    const cnt = overview.splits.counts?.[name];
                    const total = Object.values(overview.splits.counts || {}).reduce((a, b) => a + b, 0);
                    const rate = overview.splits.positive_rates?.[name]?.failure_in_next_10min;
                    return (
                      <tr key={name}>
                        <td style={{ textTransform: "capitalize", fontWeight: 600 }}>{name}</td>
                        <td className="mono">{num(cnt)} ({total ? ((cnt / total) * 100).toFixed(1) : "—"}%)</td>
                        <td className="mono" style={{ color: rate > 0.1 ? "var(--warn)" : "var(--ok)" }}>
                          {rate != null ? (rate * 100).toFixed(2) + "%" : "—"}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
              <div className="flex-center" style={{ flexWrap: "wrap", gap: 6 }}>
                {(overview.services || []).map(s => (
                  <span key={s} className="risk-badge normal">{s}</span>
                ))}
              </div>
            </>
          )}
        </Panel>
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════
// RISK TAB
// ═══════════════════════════════════════════════════════════════
function RiskTab({ data }) {
  const services = data.risk?.services ?? [];
  const asOf = data.risk?.as_of;

  return (
    <div className="page-enter">
      <div className="section-header">
        <span className="section-title">Live Risk Scores</span>
        <span className="text-muted text-xs mono">as of {shortTs(asOf)}</span>
      </div>
      <Panel title="Service Risk Table" icon="⚠">
        {services.length === 0 ? (
          <Empty icon="📡" msg="No services found. Ensure risk engine is trained and data exists." />
        ) : (
          <table className="service-table">
            <thead>
              <tr>
                <th>Service</th>
                <th>Risk Score</th>
                <th>Fail Probability</th>
                <th>Anomaly Score</th>
                <th>Error Rate</th>
                <th>Latency Trend</th>
                <th>Level</th>
                <th>Flagged</th>
              </tr>
            </thead>
            <tbody>
              {services.map(svc => (
                <tr key={svc.service}>
                  <td style={{ fontWeight: 600, fontSize: 13 }}>
                    <span className={`status-pulse ${svc.risk ? "bad" : "ok"}`} />
                    {svc.service}
                  </td>
                  <td style={{ minWidth: 140 }}>
                    <ScoreBar value={svc.risk_score} />
                  </td>
                  <td className="mono text-sm">{pct(svc.failure_probability)}</td>
                  <td style={{ minWidth: 100 }}>
                    {svc.anomaly_score != null
                      ? <ScoreBar value={svc.anomaly_score} color="var(--crit)" />
                      : <span className="text-muted">—</span>
                    }
                  </td>
                  <td className="mono text-sm" style={{ color: svc.error_rate > 0.02 ? "var(--bad)" : "inherit" }}>
                    {svc.error_rate != null ? pct(svc.error_rate) : "—"}
                  </td>
                  <td className="mono text-sm">{svc.latency_trend != null ? f2(svc.latency_trend) + " ms" : "—"}</td>
                  <td><RiskBadge level={svc.risk_level} /></td>
                  <td>{svc.risk ? <span className="text-bad">YES</span> : <span className="text-ok">—</span>}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </Panel>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════
// INCIDENTS / RCA TAB
// ═══════════════════════════════════════════════════════════════
function IncidentsTab({ toast }) {
  const [history, setHistory] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    apiFetch("/rca/history?limit=20")
      .then(d => setHistory(d.records || []))
      .catch(() => setHistory([]))
      .finally(() => setLoading(false));
  }, []);

  const records = history || [];

  return (
    <div className="page-enter">
      <div className="section-header">
        <span className="section-title">RCA Incident History</span>
        <span className="text-muted text-xs">{records.length} records</span>
      </div>

      {loading ? (
        <Panel title="Loading…" icon="⏳">
          {[1, 2, 3].map(i => <Skel key={i} h={80} mb={10} />)}
        </Panel>
      ) : records.length === 0 ? (
        <Empty icon="🔎" msg="No RCA records yet. Submit telemetry rows to /api/rca to generate incidents." />
      ) : (
        <div className="timeline">
          {records.slice().reverse().map((rec, idx) => {
            const rca = rec.rca || rec.finding || {};
            const pkg = rec.evidence || rec.package || {};
            const service = pkg.service || rca.service || "unknown";
            const ts = rec.requested_at || rec.timestamp || "";
            const category = rca.root_cause_category || "—";
            const severity = (rca.severity || "").toLowerCase();
            const dotColor = severity === "critical" ? "var(--bad)" : severity === "high" ? "var(--warn)" : "var(--ok)";
            return (
              <div key={idx} className="timeline-item">
                <div className="timeline-dot-col">
                  <div className="timeline-dot" style={{ background: dotColor, borderColor: dotColor }} />
                  <div className="timeline-line" />
                </div>
                <div className="timeline-content">
                  <div className="rca-meta">
                    <span className="rca-service">{service}</span>
                    <span className="rca-ts">{shortTs(ts)} · {relTime(ts)}</span>
                    <span className="rca-category">{category}</span>
                    {rca.severity && (
                      <span className={`risk-badge ${
                        severity === "critical" ? "critical" :
                        severity === "high" ? "warning" : "normal"
                      }`}>{rca.severity}</span>
                    )}
                  </div>
                  {rca.explanation && (
                    <div className="rca-body" style={{ marginBottom: 6 }}>
                      <strong>Explanation: </strong>{rca.explanation}
                    </div>
                  )}
                  {rca.recommended_action && (
                    <div className="rca-body">
                      <strong>Recommended: </strong>{rca.recommended_action}
                    </div>
                  )}
                  {rca.confidence_score != null && (
                    <div className="mt-8 flex-center">
                      <span className="text-muted text-xs">Confidence</span>
                      <ScoreBar value={rca.confidence_score} color="var(--accent)" />
                    </div>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════
// REMEDIATION TAB
// ═══════════════════════════════════════════════════════════════
function RemediationTab({ toast }) {
  const [pending, setPending] = useState(null);
  const [recHistory, setRecHistory] = useState(null);
  const [loadingApprove, setLoadingApprove] = useState({});
  const [activeTab, setActiveTab] = useState("pending");

  const loadPending = useCallback(() => {
    apiFetch("/remediate/pending").then(d => setPending(d.pending || [])).catch(() => setPending([]));
    apiFetch("/recovery/history?limit=15").then(d => setRecHistory(d.records || [])).catch(() => setRecHistory([]));
  }, []);

  useEffect(() => { loadPending(); }, [loadPending]);

  async function approve(id) {
    setLoadingApprove(prev => ({ ...prev, [id]: true }));
    try {
      await apiFetch("/remediate/approve", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ approval_id: id }),
      });
      toast("Remediation approved and executed ✓", "success");
      loadPending();
    } catch (e) {
      toast("Approval failed: " + e.message, "error");
    } finally {
      setLoadingApprove(prev => ({ ...prev, [id]: false }));
    }
  }

  const riskColor = r => ({ low: "low", medium: "medium", high: "high" }[r] || "medium");

  return (
    <div className="page-enter">
      <div className="tabs">
        <button className={`tab-btn ${activeTab === "pending" ? "active" : ""}`} onClick={() => setActiveTab("pending")}>
          ⏳ Pending Approvals {pending?.length ? `(${pending.length})` : ""}
        </button>
        <button className={`tab-btn ${activeTab === "history" ? "active" : ""}`} onClick={() => setActiveTab("history")}>
          📋 Recovery History
        </button>
      </div>

      {activeTab === "pending" && (
        pending == null ? (
          [1,2].map(i => <Skel key={i} h={120} mb={12} />)
        ) : pending.length === 0 ? (
          <Empty icon="✅" msg="No pending approvals. System operating normally." />
        ) : (
          pending.map(p => (
            <div key={p.approval_id} className="pending-card">
              <div className="pending-header">
                <div>
                  <div className="pending-service">{p.service}</div>
                  <div className="pending-category text-muted">{p.category}</div>
                </div>
                <span className="mono text-xs text-muted">{p.approval_id?.slice(0, 8)}…</span>
              </div>
              <div className="action-list">
                {(p.actions || []).map((a, i) => (
                  <div key={i} className="action-chip">
                    <span>🔧</span>
                    <span style={{ fontWeight: 600 }}>{a.name || a}</span>
                    {a.description && <span className="text-muted text-xs">— {a.description}</span>}
                    {a.risk && <span className={`action-risk ${riskColor(a.risk)}`}>{a.risk}</span>}
                  </div>
                ))}
              </div>
              <div className="btn-row">
                <button
                  className="btn btn-success"
                  disabled={loadingApprove[p.approval_id]}
                  onClick={() => approve(p.approval_id)}
                  id={`approve-${p.approval_id}`}
                >
                  {loadingApprove[p.approval_id] ? "Approving…" : "✓ Approve & Execute"}
                </button>
                <button className="btn btn-ghost">👁 Review Plan</button>
              </div>
            </div>
          ))
        )
      )}

      {activeTab === "history" && (
        recHistory == null ? (
          [1,2,3].map(i => <Skel key={i} h={80} mb={10} />)
        ) : recHistory.length === 0 ? (
          <Empty icon="📋" msg="No recovery checks recorded yet." />
        ) : (
          <table className="service-table">
            <thead>
              <tr>
                <th>Service</th>
                <th>Outcome</th>
                <th>Action</th>
                <th>Recovery Time</th>
                <th>Checked At</th>
              </tr>
            </thead>
            <tbody>
              {recHistory.map((r, i) => (
                <tr key={i}>
                  <td style={{ fontWeight: 600 }}>{r.service}</td>
                  <td>
                    <span className={`risk-badge ${r.outcome === "SUCCESS" ? "normal" : "critical"}`}>
                      {r.outcome === "SUCCESS" ? "✓" : "✕"} {r.outcome}
                    </span>
                  </td>
                  <td className="text-muted text-sm">{r.action || "—"}</td>
                  <td className="mono text-sm">{r.recovery_time_sec != null ? `${r.recovery_time_sec}s` : "—"}</td>
                  <td className="mono text-xs text-muted">{shortTs(r.checked_at)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )
      )}
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════
// DRIFT / MODEL MONITORING TAB
// ═══════════════════════════════════════════════════════════════
function DriftTab({ drift }) {
  if (!drift) {
    return (
      <div className="page-enter">
        <Empty icon="📡" msg="Drift report not available. Run scripts/detect_drift.py to generate one." />
      </div>
    );
  }

  const dd = drift.data_drift || {};
  const perf = drift.performance || {};
  const trig = drift.trigger || {};
  const feats = Object.values(drift.feature_metrics || {}).sort((a, b) => b.psi - a.psi);
  const rollingM = perf.rolling_metrics;

  return (
    <div className="page-enter">

      {/* Retraining Alert */}
      {trig.retraining_required && (
        <div className={`retrain-alert ${trig.severity === "critical" ? "critical" : ""}`}>
          <div className="retrain-icon">{trig.severity === "critical" ? "🚨" : "⚠️"}</div>
          <div className="retrain-body">
            <div className="retrain-title">
              Retraining Required — Severity: {(trig.severity || "warning").toUpperCase()}
            </div>
            <ul className="retrain-reasons">
              {(trig.trigger_reasons || []).map((r, i) => <li key={i}>{r}</li>)}
            </ul>
          </div>
        </div>
      )}

      <div className="two-col">
        {/* Dataset Drift Summary */}
        <Panel title="Data Drift Summary" icon="🔬">
          <div className="flex-between mb-16">
            <div>
              <div className="kpi-label">Drifted Features</div>
              <div className="kpi-value" style={{ fontSize: 36 }}>
                {dd.drifted_features_count}<span style={{ fontSize: 18, color: "var(--muted)" }}>/{dd.total_features}</span>
              </div>
            </div>
            <div style={{ textAlign: "right" }}>
              <div className="kpi-label">Dataset Drift</div>
              <div style={{ fontSize: 20, fontWeight: 700, color: dd.dataset_drift ? "var(--bad)" : "var(--ok)" }}>
                {dd.dataset_drift ? "⚠ Detected" : "✓ Stable"}
              </div>
              <div className="text-muted text-xs">{pct(dd.drift_share)} of features drifted</div>
            </div>
          </div>
          <ScoreBar value={dd.drift_share || 0} color={dd.dataset_drift ? "var(--bad)" : "var(--ok)"} />
          {dd.drifted_features?.length > 0 && (
            <div className="mt-16">
              <div className="text-xs text-muted mb-8">Significantly Drifted Features</div>
              <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
                {dd.drifted_features.slice(0, 12).map(f => (
                  <span key={f} className="drift-badge significant">{f}</span>
                ))}
                {dd.drifted_features.length > 12 && (
                  <span className="text-muted text-xs">+{dd.drifted_features.length - 12} more</span>
                )}
              </div>
            </div>
          )}
        </Panel>

        {/* Model Performance Panel */}
        <Panel title="Model Performance Monitor" icon="🤖">
          {!rollingM ? (
            <div className="text-muted text-sm">Performance monitoring requires labeled data. See detect_drift.py.</div>
          ) : (
            <>
              <div className="perf-grid mb-16">
                {[
                  { label: "F1 Score", value: f4(rollingM.f1), color: rollingM.f1 > 0.7 ? "var(--ok)" : "var(--bad)" },
                  { label: "Precision", value: f4(rollingM.precision), color: rollingM.precision > 0.7 ? "var(--ok)" : "var(--warn)" },
                  { label: "Recall", value: f4(rollingM.recall), color: rollingM.recall > 0.7 ? "var(--ok)" : "var(--warn)" },
                  { label: "Accuracy", value: f4(rollingM.accuracy), color: rollingM.accuracy > 0.85 ? "var(--ok)" : "var(--warn)" },
                ].map(m => (
                  <div key={m.label} className="perf-box">
                    <div className="perf-label">{m.label}</div>
                    <div className="perf-value" style={{ color: m.color }}>{m.value}</div>
                  </div>
                ))}
              </div>
              {perf.baseline_f1 != null && (
                <div style={{ background: "var(--panel-2)", borderRadius: 8, padding: 12 }}>
                  <div className="flex-between">
                    <span className="text-muted text-xs">Baseline F1</span>
                    <span className="mono text-sm">{f4(perf.baseline_f1)}</span>
                  </div>
                  <div className="flex-between mt-8">
                    <span className="text-muted text-xs">F1 Drop (Δ)</span>
                    <span className="mono text-sm" style={{ color: (perf.f1_drop || 0) > 0.1 ? "var(--bad)" : "var(--ok)" }}>
                      {perf.f1_drop != null ? (perf.f1_drop > 0 ? "−" : "+") + f4(Math.abs(perf.f1_drop)) : "—"}
                    </span>
                  </div>
                  <div className="flex-between mt-8">
                    <span className="text-muted text-xs">Performance Status</span>
                    <span className={`risk-badge ${perf.degraded ? "critical" : "normal"}`}>
                      {perf.degraded ? "⚠ Degraded" : "✓ Healthy"}
                    </span>
                  </div>
                  <div className="flex-between mt-8">
                    <span className="text-muted text-xs">Evaluated Samples</span>
                    <span className="mono text-sm">{num(rollingM.sample_count)}</span>
                  </div>
                </div>
              )}
            </>
          )}
        </Panel>
      </div>

      {/* Feature Drift Table */}
      <Panel title={`Feature Drift Analysis — ${feats.length} features`} icon="📈">
        <div className="drift-header-row">
          <span>Feature</span>
          <span>PSI Distribution</span>
          <span style={{ textAlign: "right" }}>PSI</span>
          <span style={{ textAlign: "right" }}>KS-Stat</span>
          <span style={{ textAlign: "right" }}>Level</span>
        </div>
        <div className="drift-scroll">
          {feats.map(f => (
            <div key={f.feature} className="drift-feature-row">
              <span className="drift-feature-name" title={f.feature}>{f.feature}</span>
              <ScoreBar value={f.psi} max={Math.max(...feats.map(x => x.psi), 1)}
                color={f.drift_level === "significant" ? "var(--bad)" : f.drift_level === "moderate" ? "var(--warn)" : "var(--ok)"} />
              <span className="drift-mono">{f.psi.toFixed(4)}</span>
              <span className="drift-mono">{f.ks_statistic.toFixed(4)}</span>
              <DriftBadge level={f.drift_level} />
            </div>
          ))}
        </div>
      </Panel>

      {/* Recommendations */}
      {trig.recommendations?.length > 0 && (
        <Panel title="Recommendations" icon="💡">
          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            {trig.recommendations.map((r, i) => (
              <div key={i} className="action-chip">
                <span>→</span>
                <span style={{ fontSize: 12 }}>{r}</span>
              </div>
            ))}
          </div>
        </Panel>
      )}
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════
// MAIN APP
// ═══════════════════════════════════════════════════════════════
function App() {
  const [tab, setTab] = useState("overview");
  const [spinning, setSpinning] = useState(false);
  const [connected, setConnected] = useState(null); // null=loading, true, false
  const { toasts, add: toast } = useToasts();

  const [data, setData] = useState({
    health: null,
    overview: null,
    risk: null,
  });
  const [drift, setDrift] = useState(null);

  const fetchAll = useCallback(async () => {
    setSpinning(true);
    try {
      const [health, overview, risk] = await Promise.all([
        apiFetch("/health"),
        apiFetch("/overview"),
        apiFetch("/risk/current?minutes=20").catch(() => null),
      ]);
      setData({ health, overview, risk });
      setConnected(true);
    } catch (err) {
      setConnected(false);
    } finally {
      setSpinning(false);
    }
  }, []);

  const fetchDrift = useCallback(async () => {
    try {
      const d = await apiFetch("/mlops/drift");
      setDrift(d);
    } catch {
      // drift report may not exist yet
    }
  }, []);

  useEffect(() => {
    fetchAll();
    fetchDrift();
    const timer = setInterval(() => {
      fetchAll();
      if (tab === "drift") fetchDrift();
    }, REFRESH_MS);
    return () => clearInterval(timer);
  }, [fetchAll, fetchDrift, tab]);

  const handleRefresh = () => {
    fetchAll();
    if (tab === "drift") fetchDrift();
    toast("Refreshing data…", "info");
  };

  const TABS = [
    { id: "overview",   label: "Overview",    icon: "🏠" },
    { id: "risk",       label: "Risk Scores", icon: "⚠" },
    { id: "incidents",  label: "Incidents",   icon: "🔥" },
    { id: "remediate",  label: "Remediation", icon: "🔧" },
    { id: "drift",      label: "Drift & ML",  icon: "📈" },
  ];

  return (
    <div className="app-shell">
      {/* Topbar */}
      <header className="topbar">
        <div className="brand">
          <div className="brand-icon">⚡</div>
          SentinelAI
        </div>

        <nav className="nav">
          {TABS.map(t => (
            <button
              key={t.id}
              id={`nav-${t.id}`}
              className={`nav-btn ${tab === t.id ? "active" : ""}`}
              onClick={() => setTab(t.id)}
            >
              <span className="icon">{t.icon}</span>
              {t.label}
            </button>
          ))}
        </nav>

        <div className="topbar-right">
          <div className={`conn-badge ${connected === true ? "connected" : connected === false ? "error" : ""}`}>
            <div className="conn-dot" />
            {connected === null ? "Connecting…" : connected ? "Live" : "Backend offline"}
          </div>
          <button
            id="refresh-btn"
            className={`refresh-btn ${spinning ? "spinning" : ""}`}
            onClick={handleRefresh}
            title="Refresh data"
          >
            ↻
          </button>
        </div>
      </header>

      {/* Main Content */}
      <main className="content">
        {tab === "overview"  && <OverviewTab data={data} drift={drift} />}
        {tab === "risk"      && <RiskTab data={data} />}
        {tab === "incidents" && <IncidentsTab toast={toast} />}
        {tab === "remediate" && <RemediationTab toast={toast} />}
        {tab === "drift"     && <DriftTab drift={drift} />}
      </main>

      {/* Footer */}
      <footer className="footer">
        <span>SentinelAI · Predictive Incident Detection</span>
        <span className="mono">v0.6.0 — Day 23</span>
      </footer>

      <ToastContainer toasts={toasts} />
    </div>
  );
}

// Mount
const root = ReactDOM.createRoot(document.getElementById("root"));
root.render(<App />);
