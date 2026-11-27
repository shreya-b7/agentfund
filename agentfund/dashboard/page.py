"""The dashboard's single HTML page."""

PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>agentfund dashboard</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
  :root {
    --bg: #f6f7f9; --panel: #ffffff; --text: #16181d; --muted: #646b78; --border: #e2e5ea;
    --good: #12805c; --bad: #c2362f; --warn: #a76100; --accent: #2f5bd3; --badge-bg: #fff3d6; --badge-text: #7a4b00;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #0f1115; --panel: #171a21; --text: #e8eaee; --muted: #9aa1ad; --border: #2a2f3a;
      --good: #3ccf91; --bad: #ff6b62; --warn: #f0b44c; --accent: #7aa2ff; --badge-bg: #3a2c0c; --badge-text: #f5cf7a;
    }
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--text);
         font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
  header { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; padding: 18px 24px;
           border-bottom: 1px solid var(--border); background: var(--panel); }
  h1 { font-size: 18px; margin: 0; }
  h2 { font-size: 14px; margin: 0 0 12px; color: var(--muted); font-weight: 600; text-transform: uppercase; letter-spacing: .04em; }
  .badge { background: var(--badge-bg); color: var(--badge-text); padding: 3px 10px; border-radius: 999px; font-weight: 600; font-size: 12px; }
  .live { margin-left: auto; color: var(--muted); font-size: 12px; }
  .dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: var(--muted); margin-right: 6px; }
  .dot.on { background: var(--good); }
  main { padding: 20px 24px; display: grid; gap: 16px; max-width: 1280px; margin: 0 auto; }
  .cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; }
  .card, .panel { background: var(--panel); border: 1px solid var(--border); border-radius: 10px; padding: 14px 16px; }
  .card .label { color: var(--muted); font-size: 12px; }
  .card .value { font-size: 22px; font-weight: 650; margin-top: 4px; font-variant-numeric: tabular-nums; }
  .grid2 { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 16px; align-items: start; }
  .card, .panel, main > * { min-width: 0; }
  #positions { max-height: 340px; overflow-y: auto; }
  @media (max-width: 900px) { .grid2 { grid-template-columns: minmax(0, 1fr); } main { padding: 16px; } header { padding: 14px 16px; } }
  table { width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; }
  th, td { text-align: right; padding: 6px 8px; border-bottom: 1px solid var(--border); white-space: nowrap; }
  th:first-child, td:first-child { text-align: left; }
  th { color: var(--muted); font-weight: 600; font-size: 12px; }
  .scroll { overflow-x: auto; }
  .good { color: var(--good); } .bad { color: var(--bad); } .warn { color: var(--warn); } .muted { color: var(--muted); }
  .feed { max-height: 420px; overflow-y: auto; display: grid; gap: 8px; }
  .item { border: 1px solid var(--border); border-radius: 8px; padding: 8px 10px; }
  .item .top { display: flex; gap: 8px; align-items: baseline; flex-wrap: wrap; }
  .item .kind { font-weight: 600; }
  .empty { color: var(--muted); padding: 12px 0; }
  .chartbox { position: relative; height: 300px; }
  .status { background: var(--panel); border: 1px solid var(--border); border-radius: 10px; padding: 12px 16px;
            display: flex; gap: 8px 20px; flex-wrap: wrap; align-items: baseline; }
  .status b { font-weight: 600; }
  .note { color: var(--muted); font-size: 12px; margin-top: 8px; }
</style>
</head>
<body>
<header>
  <h1>agentfund</h1>
  <span class="badge">PAPER TRADING &middot; simulated money</span>
  <span class="live"><span id="dot" class="dot"></span><span id="livetext">connecting&hellip;</span></span>
</header>
<main>
  <section class="status" id="status"></section>
  <section class="cards" id="cards"></section>

  <section class="grid2">
    <div class="panel">
      <h2>Paper account value</h2>
      <div class="chartbox"><canvas id="equity"></canvas></div>
      <div class="note" id="equitynote"></div>
    </div>
    <div class="panel">
      <h2>Positions</h2>
      <div class="scroll" id="positions"></div>
    </div>
  </section>

  <section class="panel">
    <h2>Backtest: every strategy, growth of $1</h2>
    <div class="chartbox"><canvas id="backtest"></canvas></div>
    <div class="scroll" id="bttable"></div>
    <div class="note" id="btnote"></div>
  </section>

  <section class="panel">
    <h2>Decision log</h2>
    <div class="note" id="logstatus"></div>
    <div class="feed" id="feed"></div>
  </section>
</main>
<script>
const $ = (id) => document.getElementById(id);
const usd = (v) => v == null ? "–" : (v < 0 ? "-$" : "$") + Math.abs(Number(v)).toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2});
const pct = (v, d = 1) => v == null ? "–" : (v * 100).toFixed(d) + "%";
const num = (v, d = 2) => v == null ? "–" : Number(v).toFixed(d);
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const COLORS = ["#2f5bd3", "#e0702f", "#12805c", "#9b3fc2", "#c2362f", "#7a7f8c"];
const START = 100000;
let equityChart, backtestChart;

function card(label, value, cls = "") {
  return `<div class="card"><div class="label">${label}</div><div class="value ${cls}">${value}</div></div>`;
}

function renderSummary(s) {
  const a = s.account;
  const log = s.log || {};
  const logText = !log.exists ? "no entries yet" : log.ok ? "verified ✓" : "TAMPERED ✗";
  const logCls = !log.exists ? "muted" : log.ok ? "good" : "bad";
  if (a) {
    const pl = a.equity - START;
    $("cards").innerHTML =
      card("Account value", usd(a.equity)) +
      card("Profit / loss", (pl >= 0 ? "+" : "") + usd(pl) + ` (${pct(pl / START, 2)})`, pl >= 0 ? "good" : "bad") +
      card("Cash", usd(a.cash)) +
      card("Positions", `${s.positions.length}` + (s.pending ? ` <span class="muted" style="font-size:13px">+${s.pending} pending</span>` : "")) +
      card("Decision log", logText, logCls);
  } else {
    $("cards").innerHTML = card("Paper account", "not started", "muted") + card("Decision log", logText, logCls);
  }

  const rows = s.positions.map(p => `<tr><td>${p.symbol}</td><td>${num(p.qty, 4)}</td><td>${usd(p.current_price)}</td>
      <td>${usd(p.market_value)}</td><td class="${p.unrealized_pl >= 0 ? "good" : "bad"}">${usd(p.unrealized_pl)}</td></tr>`).join("");
  $("positions").innerHTML = s.positions.length
    ? `<table><tr><th>Stock</th><th>Shares</th><th>Price</th><th>Value</th><th>P&amp;L</th></tr>${rows}</table>`
    : `<div class="empty">${s.pending ? `${s.pending} orders waiting to fill at the next market open. They show up here after the next daily run.` : "No positions yet. Run <code>python -m agentfund daily</code> after the close."}</div>`;

  const hist = s.equity_history || [];
  $("equitynote").textContent = s.prices_to ? `Prices up to ${s.prices_to}. One point per day you run the paper commands.` : "";
  const data = {labels: hist.map(h => h.date), datasets: [{label: "Account value", data: hist.map(h => h.equity),
    borderColor: COLORS[0], backgroundColor: COLORS[0], pointRadius: hist.length < 30 ? 3 : 0, tension: 0.2}]};
  if (!equityChart) {
    equityChart = new Chart($("equity"), {type: "line", data, options: baseOptions(v => usd(v).split(".")[0])});
  } else { equityChart.data = data; equityChart.update(); }

  const lastDaily = (s.decisions || []).find(d => d.kind === "paper_daily");
  const lastTxt = lastDaily ? `${lastDaily.payload.message} <span class="muted">(${(lastDaily.logged_at || "").slice(0, 16).replace("T", " ")} UTC)</span>`
                            : `<span class="muted">No daily run yet. Run <code>python -m agentfund daily</code> after the close.</span>`;
  $("status").innerHTML = `<span><b>Strategy:</b> ${s.strategy || "risk_parity (default)"}</span>
    <span><b>Trading:</b> ${s.paused ? '<span class="warn">paused</span>' : '<span class="good">active</span>'}</span>
    <span><b>Last daily run:</b> ${lastTxt}</span>`;
  $("logstatus").textContent = log.exists ? log.message : "Nothing logged yet.";
  const shown = (s.decisions || []).filter(d => d.kind !== "paper_order").slice(0, 25);
  $("feed").innerHTML = shown.map(renderDecision).join("") || `<div class="empty">No decisions yet.</div>`;
}

function renderDecision(rec) {
  const p = rec.payload || {};
  const when = (rec.logged_at || "").replace("T", " ").slice(0, 19) + " UTC";
  let body = "";
  if (rec.kind === "paper_cycle") {
    const r = p.risk || {};
    const vcls = r.verdict === "approved" ? "good" : r.verdict === "vetoed" ? "bad" : "warn";
    const breaches = (r.breaches || []).map(b => `${b.rule}${b.ticker ? " " + b.ticker : ""}`).join(", ");
    body = `<span class="kind">Rebalance</span> <span class="muted">${p.strategy || ""}, prices to ${p.as_of}</span>
      <span class="${vcls}">risk: ${r.verdict}</span>
      <span class="muted">${(p.orders || []).length} orders${p.submit ? " placed" : " (dry run)"}${breaches ? " · adjusted: " + breaches : ""}</span>`;
  } else if (rec.kind === "paper_daily") {
    const cls = p.action === "kill_switch" ? "bad" : p.action === "paused" ? "warn" : p.action === "rebalanced" ? "good" : "muted";
    body = `<span class="kind">Daily run</span> <span class="muted">prices to ${p.prices_to} · ${usd(p.equity)}</span>
      <span class="${cls}">${p.action.replace("_", " ")}</span>${p.fills ? ` <span class="muted">· ${p.fills} fills</span>` : ""}`;
  } else if (rec.kind === "paper_pause") {
    body = `<span class="kind">${p.paused ? "Trading paused" : "Trading resumed"}</span>`;
  } else if (rec.kind === "paper_order") {
    const o = p.order || {};
    body = `<span class="kind">${(o.side || "").toUpperCase()} ${o.symbol}</span>
      <span class="muted">${o.notional != null ? usd(o.notional) : o.qty + " sh"} · ${p.status}</span>`;
  } else {
    body = `<span class="kind">${rec.kind}</span>`;
  }
  return `<div class="item"><div class="top">${body}</div><div class="muted" style="font-size:12px">#${rec.seq} · ${when} · ${rec.hash.slice(0, 12)}</div></div>`;
}

function baseOptions(fmt, log = false) {
  const grid = css("--border"), text = css("--muted");
  return {responsive: true, maintainAspectRatio: false, animation: false, interaction: {mode: "index", intersect: false},
    plugins: {legend: {labels: {color: text, boxWidth: 12}}},
    scales: {x: {ticks: {color: text, maxTicksLimit: 8}, grid: {color: grid}},
             y: {type: log ? "logarithmic" : "linear", ticks: {color: text, callback: fmt}, grid: {color: grid}}}};
}

async function loadBacktests() {
  const r = await (await fetch("/api/backtests")).json();
  if (r.status === "computing") { $("btnote").textContent = "Running backtests… (a few seconds)"; setTimeout(loadBacktests, 2000); return; }
  if (r.status === "error") { $("btnote").textContent = "Backtest failed: " + r.error; return; }
  const names = Object.keys(r.curves);
  const labels = r.curves[names[0]].dates;
  backtestChart = new Chart($("backtest"), {type: "line",
    data: {labels, datasets: names.map((n, i) => ({label: n, data: r.curves[n].values, borderColor: COLORS[i % COLORS.length],
      backgroundColor: COLORS[i % COLORS.length], pointRadius: 0, borderWidth: 1.6}))},
    options: baseOptions(v => "$" + Number(v).toFixed(2))});
  const rows = names.map(n => { const t = r.table[n]; return `<tr><td>${n}</td><td>${pct(t.cagr)}</td><td>${pct(t.vol)}</td>
    <td>${num(t.sharpe)}</td><td>${pct(t.max_drawdown)}</td><td>${num(t.annual_turnover)}</td><td>${num(t.cost_drag_bps_per_year, 0)} bps</td>
    <td>${pct(t.deflated_sharpe)}</td></tr>`; }).join("");
  $("bttable").innerHTML = `<table><tr><th>Strategy</th><th>Return/yr</th><th>Volatility</th><th>Sharpe</th><th>Max drawdown</th>
    <th>Turnover/yr</th><th>Costs/yr</th><th>Deflated Sharpe</th></tr>${rows}</table>`;
  $("btnote").textContent = `Data: ${r.source}. Past results on today's large caps are flattered by survivorship bias.`;
}

async function refresh() {
  try { renderSummary(await (await fetch("/api/summary")).json()); } catch (e) { console.error(e); }
}

function connect() {
  const es = new EventSource("/api/stream");
  es.onopen = () => { $("dot").classList.add("on"); $("livetext").textContent = "live: updates as decisions are logged"; };
  es.addEventListener("decision", () => refresh());
  es.onerror = () => { $("dot").classList.remove("on"); $("livetext").textContent = "reconnecting…"; };
}

refresh(); loadBacktests(); connect();
</script>
</body>
</html>
"""
