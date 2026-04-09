"""
Generate a static HTML dashboard from a clear_results.json file.

Usage:
    python generate_static_dashboard.py <path_to_clear_results.json> [output.html]

Produces a self-contained HTML file with embedded data that displays:
1. Metadata: number of traces, agents, interactions
2. Workflow graph with agent usage and transitions
3. Per-agent discovered issues tables
"""

import json
import sys
from pathlib import Path
import logging
logger = logging.getLogger(__name__)

# ─── Constants (from agentic_workflow_dashboard.py) ──────────────────────
NO_ISSUE = "No Issues"


# ─── Data extraction functions ───────────────────────────────────────────
def _maybe_parse_json_text(value):
    """Parse JSON-looking string values into Python objects recursively when possible."""
    if value is None:
        return value
    if isinstance(value, list):
        return [_maybe_parse_json_text(item) for item in value]
    if isinstance(value, dict):
        return {key: _maybe_parse_json_text(val) for key, val in value.items()}
    if not isinstance(value, str):
        return value

    text = value.strip()
    if text and text[0] in "[{":
        try:
            parsed = json.loads(text)
            return _maybe_parse_json_text(parsed)
        except (json.JSONDecodeError, TypeError):
            return value
    return value


def load_json_data(json_path, include_examples=True):
    """Load all dashboard data from clear_results.json and map it to the HTML data shape."""
    json_path = Path(json_path)

    raw_data = json.loads(json_path.read_text(encoding="utf-8"))
    metadata = raw_data.get("metadata", {})
    stats = metadata.get("statistics", {})
    workflow = metadata.get("workflow_graph", metadata.get("workflow", {}))
    raw_node_stats = workflow.get("node_stats", {})
    edges = workflow.get("edges", [])

    node_stats = {}
    for agent_name, agent_stats in raw_node_stats.items():
        tasks = agent_stats.get("tasks", [])
        unique_tasks = len(tasks) if isinstance(tasks, list) else int(agent_stats.get("unique_tasks", 0))
        node_stats[agent_name] = {
            "count": int(agent_stats.get("count", 0)),
            "unique_tasks": unique_tasks,
        }

    agents_data = {}
    for agent_name, agent_payload in raw_data.get("agents", {}).items():
        summary = agent_payload.get("agent_summary", {})
        issues = agent_payload.get("issues", [])
        no_issues = agent_payload.get("no_issues", [])

        issues_table = []
        for issue in issues:
            issue_text = issue.get("issue_text") or issue.get("issue_id") or ""
            examples = []
            if include_examples:
                for occurrence in issue.get("occurrences", [])[:3]:
                    span_reference = occurrence.get("span_reference", {})
                    input_output_pair = occurrence.get("input_output_pair", {})
                    evaluation = occurrence.get("evaluation", {})
                    examples.append({
                        "trace_id": occurrence.get("trace_id", ""),
                        "step_in_trace": span_reference.get("step_in_trace", ""),
                        "model_input": _maybe_parse_json_text(input_output_pair.get("model_input", "")),
                        "response": _maybe_parse_json_text(input_output_pair.get("response", "")),
                        "score": input_output_pair.get("score", ""),
                        "evaluation_summary": _maybe_parse_json_text(evaluation.get("evaluation_summary", "")),
                    })

            issues_table.append({
                "issue": issue_text,
                "count": int(issue.get("occurrence_count", 0)),
                "freq": round(float(issue.get("frequency", 0)) * 100, 1) if isinstance(issue.get("frequency", 0), (int, float)) and float(issue.get("frequency", 0)) <= 1 else round(float(issue.get("frequency", 0)), 1),
                "severity": round(float(issue.get("severity", 0)), 2),
                "is_no_issue": False,
                "examples": examples,
            })

        if no_issues:
            total_interactions = int(summary.get("total_interactions", 0))
            no_issue_count = len(no_issues)
            no_issue_freq = round((100 * no_issue_count / total_interactions), 1) if total_interactions else 0.0
            issues_table.append({
                "issue": NO_ISSUE,
                "count": no_issue_count,
                "freq": no_issue_freq,
                "severity": 0.0,
                "is_no_issue": True,
            })

        issues_table = sorted(
            [row for row in issues_table if not row["is_no_issue"]],
            key=lambda row: -row["count"],
        ) + [row for row in issues_table if row["is_no_issue"]]

        agents_data[agent_name] = {
            "total_evals": int(summary.get("total_interactions", 0)),
            "avg_score": round(float(summary["avg_score"]), 2) if summary.get("avg_score") is not None else None,
            "unique_issues": len(issues),
            "unique_tasks": node_stats.get(agent_name, {}).get("unique_tasks", 0),
            "issues_table": issues_table,
        }

    return {
        "unique_tasks": int(stats.get("total_traces", 0)),
        "unique_agents":  len(agents_data),
        "total_rows": int(stats.get("total_interactions_analyzed", 0)),
        "node_stats": node_stats,
        "edges": edges,
        "agents": agents_data,
    }


# ─── HTML template ───────────────────────────────────────────────────────

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Agentic Workflow Dashboard</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
:root {
  --primary:#6366F1;--primary-dark:#4338CA;--primary-bg:#EEF2FF;
  --text:#1E293B;--text-secondary:#475569;--text-light:#64748B;
  --border:#E2E8F0;--page-bg:#F8FAFC;
}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Inter',-apple-system,BlinkMacSystemFont,sans-serif;background:var(--page-bg);color:var(--text);line-height:1.6}
.container{max-width:1400px;margin:0 auto;padding:24px 32px}
.dashboard-header{background:linear-gradient(135deg,#6366F1 0%,#8B5CF6 50%,#A855F7 100%);color:#fff;padding:40px 48px;border-radius:24px;margin-bottom:32px;box-shadow:0 10px 40px rgba(99,102,241,.2)}
.dashboard-header h1{font-size:32px;font-weight:700;letter-spacing:-.02em}
.dashboard-header p{margin:12px 0 0;opacity:.95;font-size:16px}
.metrics-row{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:16px;margin-bottom:32px}
.metric-card{background:#fff;border:1px solid var(--border);border-radius:16px;padding:24px 28px;box-shadow:0 1px 3px rgba(0,0,0,.06);text-align:center;transition:all .3s ease}
.metric-card:hover{box-shadow:0 10px 25px rgba(0,0,0,.1);transform:translateY(-2px)}
.metric-label{font-size:12px;font-weight:600;text-transform:uppercase;letter-spacing:.08em;color:var(--text-light);margin-bottom:8px}
.metric-value{font-size:32px;font-weight:700;color:#0F172A}
.section-header{margin:32px 0 16px}
.section-title{font-size:20px;font-weight:700;color:#0F172A;letter-spacing:-.02em}
.section-subtitle{font-size:14px;color:var(--text-secondary);margin-top:4px}
.divider{height:1px;background:var(--border);margin:24px 0}
.graph-container{background:#fff;border-radius:16px;box-shadow:0 4px 6px rgba(0,0,0,.1);overflow:hidden;margin-bottom:24px;position:relative}
.graph-canvas{width:100%;height:600px;display:block}
.graph-tooltip{display:none;position:absolute;background:#fff;border:1px solid #CBD5E1;border-radius:8px;padding:12px 16px;box-shadow:0 4px 12px rgba(0,0,0,.15);pointer-events:none;font-size:13px;z-index:100;max-width:280px}
.graph-tooltip.visible{display:block}
.graph-tooltip b{color:#0F172A}
.node-table{width:100%;border-collapse:collapse;font-size:14px;box-shadow:0 2px 8px rgba(0,0,0,.07);border-radius:10px;overflow:hidden}
.node-table thead tr{background:#6366F1;color:#fff;font-size:12px;text-transform:uppercase;letter-spacing:.05em}
.node-table th{padding:11px 16px;font-weight:600;text-align:left}
.node-table th:nth-child(2),.node-table th:nth-child(3){text-align:center}
.node-table td{padding:11px 16px}
.node-table td:nth-child(2),.node-table td:nth-child(3){text-align:center;font-weight:700;font-size:15px}
.node-table tbody tr{border-bottom:1px solid #E2E8F0}
.major-divider{margin:48px 0;padding:0;border:none;height:3px;background:linear-gradient(90deg,transparent,#6366F1,transparent);position:relative}
.major-divider::after{content:'';display:block;width:12px;height:12px;background:#6366F1;border-radius:50%;position:absolute;top:50%;left:50%;transform:translate(-50%,-50%)}
.agent-section{background:#fff;border-radius:16px;border:1px solid #1E293B;padding:24px;margin-bottom:32px}
.agent-section h3{font-size:18px;font-weight:700;color:#0F172A;margin-bottom:16px;padding-bottom:12px;border-bottom:2px solid var(--primary-bg)}
.agent-metrics-banner{background:linear-gradient(135deg,#f093fb 0%,#f5576c 100%);padding:24px;border-radius:12px;margin-bottom:24px;box-shadow:0 10px 25px rgba(240,147,251,.3)}
.agent-metrics-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:24px}
.agent-metric{text-align:center;color:#fff}
.agent-metric-label{font-size:14px;opacity:.9;margin-bottom:4px}
.agent-metric-value{font-size:32px;font-weight:700}
.issues-table{width:100%;border-collapse:collapse;font-size:14px;box-shadow:0 2px 8px rgba(0,0,0,.07);border-radius:10px;overflow:hidden}
.issues-table thead tr{background:#4F46E5;color:#fff;font-size:12px;text-transform:uppercase;letter-spacing:.05em}
.issues-table th{padding:11px 16px;font-weight:600;text-align:left}
.issues-table th:nth-child(2),.issues-table th:nth-child(3),.issues-table th:nth-child(4){text-align:center}
.issues-table td{padding:9px 16px}
.issues-table td:nth-child(2),.issues-table td:nth-child(3),.issues-table td:nth-child(4){text-align:center}
.issues-table tbody tr{border-bottom:1px solid #E2E8F0}
.issues-table tbody tr:nth-child(even){background:#F8FAFC}
.issues-table tbody tr:nth-child(odd){background:#FFF}
.severity-badge{padding:2px 10px;border-radius:999px;font-weight:600;font-size:13px;display:inline-block}
.no-issue-row{background:#F0FDF4!important;border-top:2px solid #BBF7D0!important}
.no-issue-row td:first-child{color:#15803D;font-weight:600}
.issue-main-row{cursor:pointer}
.issue-main-row:hover{background:#F1F5F9!important}
.issue-example-row td{padding:0!important;background:#F8FAFC}
.issue-examples-wrap{padding:18px 18px 22px 18px}
.issue-examples-title{font-size:15px;font-weight:800;color:#334155;text-transform:uppercase;letter-spacing:.08em;margin-bottom:14px}
.issue-examples-list{display:grid;gap:14px}
.issue-example-card{background:linear-gradient(180deg,#FFFFFF 0%,#F8FAFC 100%);border:1px solid #D8E1EE;border-radius:14px;box-shadow:0 2px 6px rgba(15,23,42,.05);overflow:hidden}
.issue-example-summary{display:flex;align-items:center;justify-content:space-between;gap:12px;padding:14px 16px;cursor:pointer;list-style:none;background:#FFFFFF}
.issue-example-summary::-webkit-details-marker{display:none}
.issue-example-summary:hover{background:#F8FAFC}
.issue-example-summary-main{display:flex;flex-direction:column;gap:4px}
.issue-example-card-title{font-size:16px;font-weight:800;color:#0F172A}
.issue-example-meta{font-size:13px;color:#64748B;line-height:1.4}
.issue-example-chevron{font-size:16px;font-weight:800;color:#6366F1;transition:transform .2s ease;flex-shrink:0}
.issue-example-card[open] .issue-example-chevron{transform:rotate(90deg)}
.issue-example-card-body{padding:18px 16px 18px 16px;border-top:1px solid #E2E8F0;background:linear-gradient(180deg,#FFFFFF 0%,#F8FAFC 100%)}
.issue-example-fields{display:grid;gap:18px}
.issue-example-field{display:grid;gap:8px}
.issue-example-label{font-size:13px;font-weight:800;color:#4F46E5;text-transform:uppercase;letter-spacing:.08em}
.issue-example-value{font-size:14px;color:#334155;white-space:pre-wrap;word-break:break-word;line-height:1.6}
.issue-example-value.code{font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,monospace;background:#0F172A;color:#E2E8F0;padding:14px 16px;border-radius:10px;overflow:auto;max-height:260px;min-height:84px}
.issue-example-scrollbox{overflow:auto;max-height:260px;min-height:84px}
.chat-message-list{display:grid;gap:12px}
.chat-message{border:1px solid #E2E8F0;border-radius:12px;padding:12px 14px;background:#fff}
.chat-message.system{background:#EEF2FF}
.chat-message.user{background:#F8FAFC}
.chat-message.assistant{background:#ECFDF5}
.chat-message-header{font-size:12px;font-weight:800;text-transform:uppercase;letter-spacing:.08em;color:#4338CA;margin-bottom:8px}
.chat-message-content{font-size:14px;color:#1E293B;white-space:pre-wrap;word-break:break-word;line-height:1.6}
.chat-message-tools{margin-top:10px;padding-top:10px;border-top:1px dashed #CBD5E1}
.expand-indicator{display:block;margin-top:8px;color:#6366F1;font-weight:800;font-size:13px}
@media(max-width:768px){.container{padding:16px}.dashboard-header{padding:24px}.agent-metrics-grid{grid-template-columns:repeat(2,1fr)}}
.info-tooltip{position:relative;display:inline-block}
.info-tooltip-trigger{display:inline-flex;align-items:center;justify-content:center;width:16px;height:16px;margin-left:6px;border-radius:50%;background:#E2E8F0;color:#475569;font-size:11px;font-weight:700;cursor:help;vertical-align:middle}
.info-tooltip-bubble{visibility:hidden;opacity:0;position:absolute;left:calc(100% + 8px);top:50%;transform:translateY(-50%);width:280px;background:#0F172A;color:#fff;padding:10px 12px;border-radius:8px;font-size:12px;line-height:1.4;box-shadow:0 8px 24px rgba(15,23,42,.2);transition:opacity .2s ease;z-index:20;text-transform:none;letter-spacing:normal;font-weight:400}
.info-tooltip:hover .info-tooltip-bubble,.info-tooltip:focus-within .info-tooltip-bubble{visibility:visible;opacity:1}
</style>
</head>
<body>
<div class="container">
  <div class="dashboard-header">
    <h1>Agentic Workflow Dashboard</h1>
    <p>Explore agent traces, analyze performance, and discover patterns in multi-agent systems</p>
  </div>
  <div class="metrics-row" id="metricsRow"></div>
  <div class="divider"></div>
  <div class="section-header">
    <div class="section-title">Agent Workflow Graph</div>
    <div class="section-subtitle">Hover over nodes for details. Node size reflects call count.</div>
  </div>
  <div class="graph-container">
    <canvas class="graph-canvas" id="graphCanvas"></canvas>
    <div class="graph-tooltip" id="graphTooltip"></div>
  </div>
  <div class="divider"></div>
  <div class="section-header">
    <div class="section-title">Node Usage Frequency</div>
    <div class="section-subtitle">Number of times each node was called across all traces</div>
  </div>
  <div id="nodeTable"></div>
  <div class="major-divider"></div>
  <div class="section-header">
    <div class="section-title" style="font-size:24px">Agent CLEAR Analysis</div>
    <div class="section-subtitle">Discovered issues for each agent</div>
  </div>
  <div id="agentSections"></div>
</div>

<script>
// ─── Embedded data ──────────────────────────────────────────────────────
const DATA = __DATA_PLACEHOLDER__;

// ─── Helpers ────────────────────────────────────────────────────────────
function esc(s) { const d=document.createElement('div'); d.textContent=s; return d.innerHTML; }
function isChatMessageList(v) {
  return Array.isArray(v) && v.every(item => item && typeof item==='object' && ('role' in item || 'content' in item));
}
function renderChatMessageList(messages) {
  return `<div class="chat-message-list">${messages.map(msg => {
    const role=msg.role||'message';
    const content=msg.content===undefined||msg.content===null?'':msg.content;
    const toolCalls=msg.tool_calls;
    const toolsHtml=(Array.isArray(toolCalls) && toolCalls.length)
      ? `<div class="chat-message-tools">${renderExampleValue(toolCalls, true)}</div>`
      : '';
    return `<div class="chat-message ${esc(String(role).toLowerCase())}">
      <div class="chat-message-header">${esc(String(role))}</div>
      <div class="chat-message-content">${fmtExampleValue(content)}</div>
      ${toolsHtml}
    </div>`;
  }).join('')}</div>`;
}
function decodeEscapedString(s) {
  if(typeof s!=='string') return s;
  return s
    .replace(/\\\\/g, '\\')
    .replace(/\\"/g, '"')
    .replace(/\\n/g, '\n')
    .replace(/\\r/g, '\r')
    .replace(/\\t/g, '\t')
    .replace(/\\u([0-9a-fA-F]{4})/g, (_, hex) => String.fromCharCode(parseInt(hex, 16)));
}
function prettyJsonStringify(v) {
  return JSON.stringify(v, null, 2);
}
function fmtExampleValue(v) {
  if(v===null||v===undefined||v==='') return '—';
  if(typeof v==='object') return esc(prettyJsonStringify(v));
  return esc(decodeEscapedString(String(v)));
}
function isJsonLike(v) {
  return v!==null && v!==undefined && v!=='' && typeof v==='object';
}
function renderStructuredJson(value, depth=0) {
  if(Array.isArray(value)) {
    if(!value.length) return `<span>[ ]</span>`;
    return `<div style="padding-left:${depth ? 16 : 0}px">[
      ${value.map(item => `<div style="margin:6px 0;padding-left:16px">${renderStructuredJson(item, depth + 1)}</div>`).join('')}
    ]</div>`;
  }
  if(value && typeof value==='object') {
    const entries=Object.entries(value);
    if(!entries.length) return `<span>{ }</span>`;
    return `<div style="padding-left:${depth ? 16 : 0}px">{
      ${entries.map(([k,v]) => `<div style="margin:6px 0;padding-left:16px"><span style="color:#93C5FD">"${esc(k)}"</span>: ${renderStructuredJson(v, depth + 1)}</div>`).join('')}
    }</div>`;
  }
  if(typeof value==='string') {
    return `<span style="white-space:pre-wrap;word-break:break-word;color:#E2E8F0">"${esc(decodeEscapedString(value))}"</span>`;
  }
  if(value===null) return '<span>null</span>';
  return `<span>${esc(String(value))}</span>`;
}
function renderExampleValue(value, forceCode=false, fieldName='') {
  const shouldRenderAsJson = forceCode || isJsonLike(value) || ((fieldName === 'response' || fieldName === 'model_input') && typeof value === 'object');
  if(fieldName !== 'model_input' && isChatMessageList(value)) return renderChatMessageList(value);
  const cls=shouldRenderAsJson?'issue-example-value code':'issue-example-value';
  const rendered = shouldRenderAsJson && isJsonLike(value) ? renderStructuredJson(value) : fmtExampleValue(value);
  return `<div class="${cls}">${rendered}</div>`;
}
function renderExampleField(label, value) {
  const scrollClass = (label === 'model_input' || label === 'response') ? ' issue-example-scrollbox' : '';
  const renderedValue = renderExampleValue(value, false, label)
    .replace('issue-example-value code', `issue-example-value code${scrollClass}`)
    .replace('issue-example-value"', `issue-example-value${scrollClass}"`);
  return `<div class="issue-example-field"><div class="issue-example-label">${esc(label)}</div>${renderedValue}</div>`;
}

// ─── Metrics ────────────────────────────────────────────────────────────
document.getElementById('metricsRow').innerHTML = [
  ['#Agents <span class="info-tooltip"><span class="info-tooltip-trigger" tabindex="0" aria-label="Agents explanation">i</span><span class="info-tooltip-bubble">Number of evaluated subagent/nodes</span></span>', DATA.unique_agents],
  ['#Traces <span class="info-tooltip"><span class="info-tooltip-trigger" tabindex="0" aria-label="Traces explanation">i</span><span class="info-tooltip-bubble">Number of input traces</span></span>', DATA.unique_tasks],
  ['#Interactions <span class="info-tooltip"><span class="info-tooltip-trigger" tabindex="0" aria-label="Interactions explanation">i</span><span class="info-tooltip-bubble">Total number of evaluated llm calls</span></span>', DATA.total_rows],
].map(([l,v]) => `<div class="metric-card"><div class="metric-label">${l}</div><div class="metric-value">${esc(String(v))}</div></div>`).join('');

// ─── Graph ──────────────────────────────────────────────────────────────
function shellLayout(nodes, W, H) {
  const cx=W/2, cy=H/2, r=Math.min(W,H)*.32, n=nodes.length, pos={};
  nodes.forEach((nd,i)=>{ const a=2*Math.PI*i/n - Math.PI/2; pos[nd]={x:cx+r*Math.cos(a), y:cy+r*Math.sin(a)}; });
  return pos;
}

function renderGraph() {
  const c=document.getElementById('graphCanvas'), ctx=c.getContext('2d');
  const dpr=devicePixelRatio||1, rect=c.getBoundingClientRect();
  c.width=rect.width*dpr; c.height=rect.height*dpr;
  ctx.scale(dpr,dpr);
  const W=rect.width, H=rect.height;
  const ns=DATA.node_stats, edges=DATA.edges;
  const names=Object.keys(ns);
  const pos=shellLayout(names, W, H);
  const counts=names.map(n=>ns[n].count);
  const minC=Math.min(...counts), maxC=Math.max(...counts);
  const nr = cnt => maxC===minC ? 30 : 18+((cnt-minC)/(maxC-minC))*32;
  const nc = cnt => { if(maxC===minC) return '#A5B4FC'; const n=(cnt-minC)/(maxC-minC); return n>.7?'#4F46E5':n>.4?'#6366F1':'#818CF8'; };

  ctx.clearRect(0,0,W,H); ctx.fillStyle='#FAFBFC'; ctx.fillRect(0,0,W,H);

  const edgeSet=new Set(edges.map(e=>e.src+'|||'+e.tgt));

  // Edges
  for(const e of edges) {
    const p0=pos[e.src], p1=pos[e.tgt]; if(!p0||!p1) continue;
    let x0=p0.x, y0=p0.y, x1=p1.x, y1=p1.y;
    const dx=x1-x0, dy=y1-y0, len=Math.sqrt(dx*dx+dy*dy);
    if(edgeSet.has(e.tgt+'|||'+e.src) && len>0) {
      const px=-dy/len*14, py=dx/len*14; x0+=px; y0+=py; x1+=px; y1+=py;
    }
    const r0=nr(ns[e.src]?.count||0), r1=nr(ns[e.tgt]?.count||0);
    const la=Math.sqrt((x1-x0)**2+(y1-y0)**2);
    if(la<=r0+r1) continue;
    const dA=(x1-x0)/la, dB=(y1-y0)/la;
    const sx=x0+dA*(r0+4), sy=y0+dB*(r0+4), ex=x1-dA*(r1+4), ey=y1-dB*(r1+4);
    // Line
    const lw=Math.min(e.weight/2,6)+1;
    ctx.beginPath(); ctx.moveTo(sx,sy); ctx.lineTo(ex,ey);
    ctx.strokeStyle='#94A3B8'; ctx.lineWidth=lw; ctx.stroke();
    // Arrowhead - larger, filled, at the end of the line
    const al=14+lw, ag=Math.atan2(ey-sy,ex-sx);
    ctx.beginPath(); ctx.moveTo(ex,ey);
    ctx.lineTo(ex-al*Math.cos(ag-.4),ey-al*Math.sin(ag-.4));
    ctx.lineTo(ex-al*.5*Math.cos(ag),ey-al*.5*Math.sin(ag));
    ctx.lineTo(ex-al*Math.cos(ag+.4),ey-al*Math.sin(ag+.4));
    ctx.closePath(); ctx.fillStyle='#64748B'; ctx.fill();
    // Weight label
    const lx=(sx+ex)/2, ly=(sy+ey)/2;
    ctx.font='600 12px Inter,sans-serif';
    const tw=ctx.measureText(String(e.weight)).width;
    ctx.fillStyle='rgba(255,255,255,.95)';
    ctx.beginPath(); ctx.roundRect(lx-tw/2-6,ly-10,tw+12,20,4); ctx.fill();
    ctx.strokeStyle='#CBD5E1'; ctx.lineWidth=1; ctx.stroke();
    ctx.fillStyle='#1E293B'; ctx.textAlign='center'; ctx.textBaseline='middle';
    ctx.fillText(String(e.weight),lx,ly);
  }

  // Nodes
  for(const name of names) {
    const p=pos[name], r=nr(ns[name].count), col=nc(ns[name].count);
    ctx.save(); ctx.shadowColor='rgba(0,0,0,.12)'; ctx.shadowBlur=8; ctx.shadowOffsetY=2;
    ctx.beginPath(); ctx.arc(p.x,p.y,r,0,Math.PI*2); ctx.fillStyle=col; ctx.fill(); ctx.restore();
    ctx.beginPath(); ctx.arc(p.x,p.y,r,0,Math.PI*2); ctx.strokeStyle='#4338CA'; ctx.lineWidth=2.5; ctx.stroke();
    // Label above node only (black text)
    ctx.font='600 13px Inter,sans-serif'; ctx.fillStyle='#1E293B'; ctx.textAlign='center'; ctx.textBaseline='bottom';
    const fn=name.length>28?name.slice(0,25)+'...':name;
    ctx.fillText(fn,p.x,p.y-r-8);
  }

  // Hover data
  c._gd={pos,ns,nr};
}

// Tooltip
(function(){
  const c=document.getElementById('graphCanvas'), tt=document.getElementById('graphTooltip');
  c.addEventListener('mousemove',e=>{
    const r=c.getBoundingClientRect(), mx=e.clientX-r.left, my=e.clientY-r.top, gd=c._gd;
    if(!gd) return;
    for(const[name,p] of Object.entries(gd.pos)){
      const rad=gd.nr(gd.ns[name].count);
      if(Math.sqrt((mx-p.x)**2+(my-p.y)**2)<=rad){
        const s=gd.ns[name];
        tt.innerHTML=`<b>${esc(name)}</b><br><br><b>Total Calls:</b> ${s.count}<br><b>Unique Tasks:</b> ${s.unique_tasks}`;
        tt.style.left=(mx+16)+'px'; tt.style.top=(my+16)+'px';
        tt.classList.add('visible'); return;
      }
    }
    tt.classList.remove('visible');
  });
  c.addEventListener('mouseleave',()=>tt.classList.remove('visible'));
})();

renderGraph();
window.addEventListener('resize', renderGraph);

// ─── Heatmap color helper ───────────────────────────────────────────────
function heatBg(val, min, max) {
  // Returns a background color from light blue to deep indigo based on normalized value
  if(max===min) return 'rgba(99,102,241,0.10)';
  const t=(val-min)/(max-min);
  const r=Math.round(238 - t*139);  // 238 -> 99
  const g=Math.round(242 - t*140);  // 242 -> 102
  const b=Math.round(255 - t*14);   // 255 -> 241
  const a=(0.08 + t*0.25).toFixed(2);
  return `rgba(${99},${102},${241},${a})`;
}
function heatFg(val, min, max) {
  if(max===min) return '#334155';
  const t=(val-min)/(max-min);
  return t>0.5 ? '#312E81' : '#334155';
}

// ─── Node usage table ───────────────────────────────────────────────────
(function(){
  const ns=DATA.node_stats;
  const sorted=Object.entries(ns).sort((a,b)=>b[1].count-a[1].count);
  const counts=sorted.map(([,s])=>s.count), tasks=sorted.map(([,s])=>s.unique_tasks);
  const minC=Math.min(...counts), maxC=Math.max(...counts);
  const minT=Math.min(...tasks), maxT=Math.max(...tasks);
  let html='<table class="node-table"><thead><tr><th>Agent</th><th>Total Calls</th><th>Unique Tasks</th></tr></thead><tbody>';
  for(const[name,s] of sorted) {
    html+=`<tr><td style="color:#1E293B">${esc(name)}</td><td style="background:${heatBg(s.count,minC,maxC)};color:${heatFg(s.count,minC,maxC)}">${s.count}</td><td style="background:${heatBg(s.unique_tasks,minT,maxT)};color:${heatFg(s.unique_tasks,minT,maxT)}">${s.unique_tasks}</td></tr>`;
  }
  html+='</tbody></table>';
  document.getElementById('nodeTable').innerHTML=html;
})();

// ─── Agent sections ─────────────────────────────────────────────────────
(function(){
  const container=document.getElementById('agentSections');
  for(const[name,ad] of Object.entries(DATA.agents).sort((a,b)=>a[0].localeCompare(b[0]))) {
    const issues=ad.issues_table;
    const maxFreq = Math.max(...issues.filter(d=>!d.is_no_issue).map(d=>d.freq), 1);

    let tableHtml='';
    const regular=issues.filter(d=>!d.is_no_issue);
    const noIssue=issues.filter(d=>d.is_no_issue);

    if(regular.length||noIssue.length) {
      tableHtml='<table class="issues-table"><thead><tr><th>Issue</th><th>Count</th><th>Frequency</th><th>Severity</th></tr></thead><tbody>';
      for(const [idx,d] of regular.entries()) {
        const t=maxFreq>0?d.freq/maxFreq:0;
        let fc='#16A34A'; if(t>=.66) fc='#DC2626'; else if(t>=.33) fc='#D97706';
        let sb,sf;
        if(d.severity>=.7){sb='#FEE2E2';sf='#991B1B';}else if(d.severity>=.4){sb='#FEF9C3';sf='#854D0E';}else{sb='#D1FAE5';sf='#065F46';}
        const exampleId=`examples-${esc(name).replace(/[^a-zA-Z0-9_-]/g,'_')}-${idx}`;
        const examples=(d.examples||[]).map((ex, exampleIdx)=>`<details class="issue-example-card">
          <summary class="issue-example-summary">
            <div class="issue-example-summary-main">
              <div class="issue-example-card-title">Example ${exampleIdx + 1}</div>
              <div class="issue-example-meta">trace_id: ${fmtExampleValue(ex.trace_id)} · step: ${fmtExampleValue(ex.step_in_trace)}</div>
            </div>
            <div class="issue-example-chevron">›</div>
          </summary>
          <div class="issue-example-card-body">
            <div class="issue-example-fields">
              ${renderExampleField('trace_id', ex.trace_id)}
              ${renderExampleField('step_in_trace', ex.step_in_trace)}
              ${renderExampleField('model_input', ex.model_input)}
              ${renderExampleField('response', ex.response)}
              ${renderExampleField('score', ex.score)}
              ${renderExampleField('evaluation_summary', ex.evaluation_summary)}
            </div>
          </div>
        </details>`).join('');
        tableHtml+=`<tr class="issue-main-row" onclick="toggleIssueExamples('${exampleId}', this)">
          <td style="color:#1E293B">${esc(d.issue)}${(d.examples&&d.examples.length)?'<br><span class="expand-indicator">▸ Examples</span>':''}</td>
          <td style="font-weight:600;color:#334155">${d.count}</td>
          <td style="font-weight:700;color:${fc}">${d.freq}%</td>
          <td><span class="severity-badge" style="background:${sb};color:${sf}">${d.severity.toFixed(2)}</span></td>
        </tr>`;
        if(d.examples&&d.examples.length){
          tableHtml+=`<tr class="issue-example-row" id="${exampleId}" style="display:none"><td colspan="4">
            <div class="issue-examples-wrap">
              <div class="issue-examples-title">Examples</div>
              <div class="issue-examples-list">${examples}</div>
            </div>
          </td></tr>`;
        }
      }
      for(const d of noIssue) {
        tableHtml+=`<tr class="no-issue-row"><td>&#x2705; No Issues</td><td style="font-weight:600;color:#334155">${d.count}</td><td style="font-weight:700;color:#065F46">${d.freq}%</td><td><span class="severity-badge" style="background:#D1FAE5;color:#065F46">0.00</span></td></tr>`;
      }
      tableHtml+='</tbody></table>';
    } else {
      tableHtml='<div style="text-align:center;padding:20px;color:#10B981;">No issues discovered in this agent\'s outputs</div>';
    }

    const sec=document.createElement('div'); sec.className='agent-section';
    sec.innerHTML=`<h3>${esc(name)}</h3>
      <div class="agent-metrics-banner"><div class="agent-metrics-grid">
        <div class="agent-metric"><div class="agent-metric-label">#Evaluated Calls</div><div class="agent-metric-value">${ad.total_evals}</div></div>
        <div class="agent-metric"><div class="agent-metric-label">Avg Score</div><div class="agent-metric-value">${ad.avg_score!==null?ad.avg_score:'N/A'}</div></div>
        <div class="agent-metric"><div class="agent-metric-label">#Issues Discovered</div><div class="agent-metric-value">${ad.unique_issues}</div></div>
        <div class="agent-metric"><div class="agent-metric-label">#Traces</div><div class="agent-metric-value">${ad.unique_tasks}</div></div>
      </div></div>
      <div class="section-header" style="margin-top:0"><div class="section-title">Discovered Issues</div><div class="section-subtitle">Recurring problems identified in this agent's outputs <span class="info-tooltip"><span class="info-tooltip-trigger" tabindex="0" aria-label="Issue matching explanation">i</span><span class="info-tooltip-bubble">Each llm call can be matched to any number of discovered issues, or to no issues.</span></span></div></div>
      ${tableHtml}`;
    container.appendChild(sec);
  }
})();
function toggleIssueExamples(id, rowEl){
  const el=document.getElementById(id);
  if(!el) return;
  const isHidden=el.style.display==='none';
  el.style.display=isHidden?'table-row':'none';
  const indicator=rowEl.querySelector('.expand-indicator');
  if(indicator) indicator.textContent=isHidden?'▾ Examples':'▸ Examples';
}
</script>
</body>
</html>"""


def generate_html(json_path, output_path=None, include_examples=False):
    """Generate static HTML dashboard from a clear_results.json file."""
    data = load_json_data(json_path, include_examples=include_examples)
    logging.info(f"Generating Static HTML report  from {json_path}")

    input_path = Path(json_path)
    if output_path is None:
        output_path = input_path.with_suffix(".html")

    data_json = json.dumps(data, ensure_ascii=False)
    output_path_obj = Path(output_path)
    output_path_obj.with_suffix(".dashboard_data.json").write_text(data_json, encoding="utf-8")

    html = HTML_TEMPLATE.replace("__DATA_PLACEHOLDER__", data_json)

    output_path_obj.write_text(html, encoding="utf-8")
    logging.info(f"Static HTML report written to {output_path_obj}")
    return output_path_obj


if __name__ == "__main__":
    if len(sys.argv) < 2:
        logging.info("Usage: python generate_static_dashboard.py <clear_results.json> [output.html]")
        sys.exit(1)

    json_pth = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    generate_html(json_pth, output_path)
