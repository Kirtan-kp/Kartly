"use strict";
// Admin page. Everything from the server is put on the page as text (never as HTML), because
// ticket messages and tool arguments are written by customers.

const app = document.getElementById("app");
let session = null; // {username, role, csrf}
let view = "overview";
let timer = null;

function h(tag, attrs, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === null || v === undefined || v === false) continue;
    if (k === "class") el.className = v;
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else if (v === true) el.setAttribute(k, "");
    else el.setAttribute(k, String(v));
  }
  for (const kid of kids.flat()) {
    if (kid === null || kid === undefined || kid === false) continue;
    el.append(kid instanceof Node ? kid : document.createTextNode(String(kid)));
  }
  return el;
}

// replaceChildren does not flatten arrays (it would print "[object HTMLDivElement]"), so go through h().
const fill = (el, ...kids) => el.replaceChildren(h("div", {}, ...kids));

class AuthLost extends Error {}

async function api(method, path, body) {
  const headers = {};
  if (body !== undefined) headers["Content-Type"] = "application/json";
  if (method !== "GET" && session) headers["X-CSRF-Token"] = session.csrf;
  const res = await fetch(path, {
    method, headers, credentials: "same-origin",
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (res.status === 401 && path !== "/v1/admin/login") throw new AuthLost();
  if (res.status === 204) return null;
  let data = null;
  try { data = await res.json(); } catch (_) { /* not json */ }
  if (!res.ok) {
    let msg = data && data.detail ? data.detail : res.statusText;
    if (Array.isArray(msg)) msg = msg.map((m) => m.msg).join("; ");
    throw new Error(String(msg));
  }
  return data;
}

const ago = (iso) => {
  if (!iso) return "";
  const s = Math.max(0, (Date.now() - new Date(iso.endsWith("Z") || iso.includes("+") ? iso : iso + "Z")) / 1000);
  if (s < 90) return Math.round(s) + "s ago";
  if (s < 5400) return Math.round(s / 60) + "m ago";
  if (s < 129600) return Math.round(s / 3600) + "h ago";
  return Math.round(s / 86400) + "d ago";
};
const secs = (s) => (s === null || s === undefined ? "-" : s < 90 ? Math.round(s) + "s" : Math.round(s / 60) + "m");
const pct = (r) => (r === null || r === undefined ? "-" : Math.round(r * 100) + "%");
const json = (v) => JSON.stringify(v, null, 2);
const tone = (v) => ({ succeeded: "ok", executed: "ok", approved: "ok", drafted: "ok", sent: "ok", ok: "ok",
  failed: "bad", rejected: "bad", dead: "bad", firing: "bad", escalated: "warn", awaiting_approval: "warn",
  pending: "warn", expired: "warn", critical: "bad", warning: "warn" }[v] || "");
const chip = (v) => h("span", { class: "chip " + tone(v) }, v ?? "-");
const notice = (text, bad) => h("div", { class: "notice" + (bad ? " bad" : "") }, text);

// ---------------------------------------------------------------------------- shell

const VIEWS = [
  ["overview", "Overview"], ["tickets", "Tickets"], ["approvals", "Approvals"],
  ["dead", "Dead letters"], ["alerts", "Alerts"], ["controls", "Controls"], ["accounts", "Accounts"],
];
let counts = { approvals: 0, dead: 0, alerts: 0 };

function shell(content) {
  const items = VIEWS.filter(([id]) => id !== "accounts" || session.role === "admin");
  return h("div", { class: "shell" },
    h("nav", {},
      h("h1", {}, "Kartly admin"),
      items.map(([id, label]) => {
        const n = { approvals: counts.approvals, dead: counts.dead, alerts: counts.alerts }[id];
        return h("button", { "aria-current": view === id ? "page" : null, onclick: () => go(id) },
          label, n ? h("span", { class: "badge" }, n) : null);
      }),
      h("div", { class: "who" }, session.username + " (" + session.role + ")",
        h("div", {}, h("button", { class: "btn", onclick: signOut }, "Sign out")))),
    h("main", {}, content));
}

function go(id) { view = id; location.hash = id; render(); }

async function render() {
  clearTimeout(timer);
  if (!session) return renderLogin();
  const content = h("div", {}, "Loading…");
  app.replaceChildren(shell(content));
  try {
    const body = await views[view](content);
    if (body) content.replaceChildren(body);
    refreshCounts();
  } catch (e) {
    if (e instanceof AuthLost) { session = null; return renderLogin("Your session ended. Sign in again."); }
    content.replaceChildren(notice(e.message, true));
  }
  if (["overview", "approvals", "dead", "alerts"].includes(view)) {
    timer = setTimeout(() => { if (!document.hidden && !document.querySelector("textarea:focus")) render(); else timer = setTimeout(render, 15000); }, 15000);
  }
}

const unavailable = new Set(); // features this server does not have (queue, alerts): ask once
const soft = (path, empty) => unavailable.has(path) ? Promise.resolve(empty) :
  api("GET", path).catch((e) => { if (e instanceof AuthLost) throw e; unavailable.add(path); return empty; });

async function refreshCounts() {
  try {
    const [a, d, al] = await Promise.all([
      soft("/v1/approvals?status=pending", { approvals: [] }), soft("/v1/dead-letters", { jobs: [] }),
      soft("/v1/alerts", { alerts: [] }),
    ]);
    const next = { approvals: a.approvals.length, dead: d.jobs.length, alerts: al.alerts.filter((x) => x.status === "firing").length };
    if (JSON.stringify(next) !== JSON.stringify(counts)) {
      counts = next;
      const nav = document.querySelector("nav");
      if (nav) nav.replaceWith(shell(h("div")).querySelector("nav"));
    }
  } catch (_) { /* the page itself shows errors */ }
}

// ---------------------------------------------------------------------------- login

function renderLogin(message) {
  const user = h("input", { id: "u", autocomplete: "username", required: true, autofocus: true });
  const pass = h("input", { id: "p", type: "password", autocomplete: "current-password", required: true });
  const err = h("div", { class: "notice bad", hidden: !message }, message || "");
  const form = h("form", { class: "login", onsubmit: async (ev) => {
    ev.preventDefault();
    try {
      session = await api("POST", "/v1/admin/login", { username: user.value, password: pass.value });
      view = location.hash.slice(1) in views ? location.hash.slice(1) : "overview";
      render();
    } catch (e) { err.hidden = false; err.textContent = e.message; pass.value = ""; }
  } },
    h("h2", {}, "Kartly admin"), err,
    h("label", { for: "u" }, "Username"), user, h("label", { for: "p" }, "Password"), pass,
    h("button", { class: "btn primary", type: "submit" }, "Sign in"));
  app.replaceChildren(form);
}

async function signOut() {
  try { await api("POST", "/v1/admin/logout"); } catch (_) { /* already gone */ }
  session = null; counts = { approvals: 0, dead: 0, alerts: 0 }; renderLogin();
}

// ---------------------------------------------------------------------------- views

const card = (k, v, cls) => h("div", { class: "card " + (cls || "") }, h("div", { class: "k" }, k), h("div", { class: "v" }, v));

async function overview() {
  const windows = { 3600: "1 hour", 86400: "24 hours", 604800: "7 days" };
  const sel = h("select", { onchange: () => { overview.w = Number(sel.value); render(); } },
    Object.entries(windows).map(([v, l]) => h("option", { value: v, selected: Number(v) === (overview.w || 86400) }, l)));
  const m = await api("GET", "/v1/metrics?window_s=" + (overview.w || 86400));
  const r = m.runs, q = m.queue, ap = m.approvals;
  const rate = (x) => (x === null ? "-" : pct(x));
  return h("div", {},
    h("div", { class: "row" }, h("h2", { class: "m0" }, "Overview"), sel),
    h("div", { class: "grid" },
      card("Runs", r.total), card("Escalation rate", rate(r.escalation_rate)),
      card("Failure rate", rate(r.failure_rate), r.failure_rate > 0.1 ? "bad" : ""),
      card("Median / p95 latency", (r.latency_ms.p50 ?? "-") + " / " + (r.latency_ms.p95 ?? "-") + " ms"),
      card("Shadow cost", "$" + r.shadow_cost_usd), card("Model errors", m.llm.errors + " of " + m.llm.calls, m.llm.errors ? "warn" : ""),
      card("Approvals waiting", ap.pending, ap.pending ? "warn" : ""), card("Oldest waiting", secs(ap.oldest_pending_age_s)),
      card("Queue (queued / running)", q.queued + " / " + q.running), card("Dead letters", q.dead, q.dead ? "bad" : "")),
    h("h3", {}, "Runs by outcome"),
    h("div", {}, Object.entries(r.by_outcome).map(([k, v]) => h("span", { class: "chip" + " mr" }, k + " " + v)),
      Object.keys(r.by_outcome).length ? null : h("span", { class: "muted" }, "none in this window")),
    h("h3", {}, "Tools"),
    tableOf(["Tool", "Calls", "Errors"], Object.entries(m.tools).map(([k, v]) => [k, v.calls ?? v.total ?? "-", v.errors ?? 0])),
    h("h3", {}, "Controls"),
    h("div", {}, Object.entries(m.controls).map(([k, v]) => h("span", { class: "chip" + " mr" }, k + " = " + v))));
}

function tableOf(head, rows) {
  if (!rows.length) return h("div", { class: "muted" }, "nothing yet");
  return h("table", {}, h("thead", {}, h("tr", {}, head.map((x) => h("th", {}, x)))),
    h("tbody", {}, rows.map((row) => h("tr", {}, row.map((c) => h("td", {}, c))))));
}

const ticketState = { status: "", customer: "", cursors: [], page: null, selected: null };

async function tickets() {
  const params = new URLSearchParams({ limit: "20" });
  if (ticketState.status) params.set("status", ticketState.status);
  if (ticketState.customer) params.set("customer_id", ticketState.customer);
  const cursor = ticketState.cursors[ticketState.cursors.length - 1];
  if (cursor) params.set("before", cursor);
  const list = await api("GET", "/v1/tickets?" + params);
  const status = h("select", { onchange: () => { ticketState.status = status.value; ticketState.cursors = []; render(); } },
    ["", "received", "screened", "routed", "resolving", "validating", "sent", "drafted", "awaiting_approval", "escalated", "closed", "failed"].map((s) =>
      h("option", { value: s, selected: s === ticketState.status }, s || "any status")));
  const cust = h("input", { placeholder: "customer id", value: ticketState.customer, onkeydown: (e) => {
    if (e.key === "Enter") { ticketState.customer = cust.value.trim(); ticketState.cursors = []; render(); } } });
  const detail = h("div", {}, ticketState.selected ? "Loading…" : h("div", { class: "muted" }, "Pick a ticket."));
  const left = h("div", { class: "scroll" },
    h("div", { class: "row" }, status, cust),
    list.tickets.length ? h("table", {}, h("thead", {}, h("tr", {}, ["Ticket", "Customer", "Status", "Last run", "Created"].map((x) => h("th", {}, x)))),
      h("tbody", {}, list.tickets.map((t) => h("tr", { class: "click", onclick: () => { ticketState.selected = t.id; render(); } },
        h("td", {}, t.id.slice(0, 8)), h("td", {}, t.customer_id), h("td", {}, chip(t.status)),
        h("td", {}, t.last_run_outcome ? chip(t.last_run_outcome) : "-"), h("td", {}, ago(t.created_at)))))) : h("div", { class: "muted" }, "no tickets"),
    h("div", { class: "row" + " mt10" },
      h("button", { class: "btn", disabled: !ticketState.cursors.length, onclick: () => { ticketState.cursors.pop(); render(); } }, "Newer"),
      h("button", { class: "btn", disabled: !list.next_before, onclick: () => { ticketState.cursors.push(list.next_before); render(); } }, "Older")));
  if (ticketState.selected) loadTicket(ticketState.selected, detail);
  return h("div", {}, h("h2", {}, "Tickets"), h("div", { class: "split" }, left, detail));
}

async function loadTicket(id, into) {
  try {
    const t = await api("GET", "/v1/tickets/" + id);
    fill(into,
      h("h3", { class: "mt0" }, "Ticket " + t.id.slice(0, 8) + " · " + t.customer_id + " · ", chip(t.status)),
      t.messages.map((m) => h("div", { class: "msg " + m.role }, h("div", { class: "role" }, m.role + " · " + ago(m.created_at)), m.content)),
      h("h3", {}, "Runs"),
      t.runs.length ? t.runs.map((r) => h("div", { class: "card" + " mb8" },
        h("div", { class: "row" + " m0" }, chip(r.outcome || r.status), h("span", { class: "muted" },
          (r.model_tier || "-") + " · " + (r.tokens_in + r.tokens_out) + " tokens · $" + r.shadow_cost_usd + " · " + ago(r.started_at)),
          h("button", { class: "btn", onclick: (e) => showTrace(r.id, e.target.parentElement.parentElement) }, "Trace")),
        r.error ? h("pre", {}, r.error) : null)) : h("div", { class: "muted" }, "no runs yet"));
  } catch (e) { if (e instanceof AuthLost) { session = null; return renderLogin("Your session ended. Sign in again."); } into.replaceChildren(notice(e.message, true)); }
}

async function showTrace(runId, host) {
  let box = host.querySelector(".trace");
  if (box) { box.remove(); return; }
  box = h("div", { class: "trace" }, "Loading…"); host.append(box);
  try {
    const t = await api("GET", "/v1/runs/" + runId + "/trace");
    fill(box,
      t.steps.map((s) => h("details", {},
        h("summary", {}, "#" + s.idx + " " + s.kind + " · " + s.name + (s.model ? " · " + s.model : "") +
          (s.latency_ms !== null ? " · " + s.latency_ms + " ms" : "") + (s.error ? " · ERROR" : "")),
        s.error ? h("pre", {}, s.error) : null,
        s.input ? h("pre", {}, "input " + json(s.input)) : null, s.output ? h("pre", {}, "output " + json(s.output)) : null)),
      t.audit.length ? h("details", {}, h("summary", {}, "Audit (" + t.audit.length + ")"),
        h("pre", {}, t.audit.map((a) => a.ts + "  " + a.action + (a.details ? "  " + JSON.stringify(a.details) : "")).join("\n"))) : null);
  } catch (e) { box.replaceChildren(notice(e.message, true)); }
}

async function approvals() {
  const data = await api("GET", "/v1/approvals?status=pending");
  const done = await api("GET", "/v1/approvals?status=all&limit=15");
  const decide = async (a, verb, note, btns, out) => {
    btns.forEach((b) => (b.disabled = true));
    try {
      const r = await api("POST", "/v1/approvals/" + a.id + "/" + verb, note.value ? { note: note.value } : {});
      out.replaceChildren(notice(r.message || r.result));
      setTimeout(render, 1200);
    } catch (e) { out.replaceChildren(notice(e.message, true)); btns.forEach((b) => (b.disabled = false)); }
  };
  return h("div", {}, h("h2", {}, "Approvals"),
    data.approvals.length ? data.approvals.map((a) => {
      const note = h("textarea", { placeholder: "note (optional)", maxlength: "500" });
      const out = h("div", {});
      const approve = h("button", { class: "btn primary", onclick: () => decide(a, "approve", note, [approve, reject], out) }, "Approve");
      const reject = h("button", { class: "btn danger", onclick: () => decide(a, "reject", note, [approve, reject], out) }, "Reject");
      return h("div", { class: "card" + " mb12" },
        h("div", { class: "row" + " m0b6" }, h("b", {}, a.tool), chip(a.action_status), h("span", { class: "muted" }, a.customer_id + " · ticket " + a.ticket_id.slice(0, 8) + " · asked " + ago(a.requested_at) + " · expires " + ago(a.expires_at).replace(" ago", ""))),
        h("div", {}, "Why: ", a.reason),
        h("details", { open: true }, h("summary", {}, "What it would do"), h("pre", {}, json(a.preview || a.args))),
        h("details", {}, h("summary", {}, "Arguments"), h("pre", {}, json(a.args))),
        note, h("div", { class: "row" + " mt8" }, approve, reject), out);
    }) : notice("Nothing is waiting for approval."),
    h("h3", {}, "Recent decisions"),
    tableOf(["Tool", "Customer", "Status", "By", "When"], done.approvals.filter((a) => a.status !== "pending").reverse().map((a) =>
      [a.tool, a.customer_id, chip(a.status), a.decided_by || "-", ago(a.decided_at || a.requested_at)])));
}

async function dead() {
  let data;
  try { data = await api("GET", "/v1/dead-letters"); } catch (e) {
    if (e instanceof AuthLost) throw e;
    return h("div", {}, h("h2", {}, "Dead letters"), notice("The job queue is not enabled on this server (" + e.message + ")."));
  }
  const requeue = async (j, btn) => {
    btn.disabled = true;
    try { await api("POST", "/v1/dead-letters/" + j.job_id + "/requeue", {}); render(); } catch (e) { btn.disabled = false; btn.after(" " + e.message); }
  };
  return h("div", {}, h("h2", {}, "Dead letters"), h("div", { class: "muted" + " mb8" }, data.backlog + " job(s) waiting or running"),
    data.jobs.length ? h("table", {}, h("thead", {}, h("tr", {}, ["Ticket", "Attempts", "Last error", "Died", ""].map((x) => h("th", {}, x)))),
      h("tbody", {}, data.jobs.map((j) => { const b = h("button", { class: "btn", onclick: () => requeue(j, b) }, "Requeue");
        return h("tr", {}, h("td", {}, j.ticket_id.slice(0, 8)), h("td", {}, j.attempts), h("td", {}, h("pre", {}, j.last_error || "-")), h("td", {}, ago(j.died_at)), h("td", {}, b)); })))
      : notice("No dead letters."));
}

async function alerts() {
  const data = await api("GET", "/v1/alerts");
  const check = h("button", { class: "btn", onclick: async () => { check.disabled = true; try { await api("POST", "/v1/alerts/check"); } catch (_) { /* shown by render */ } render(); } }, "Check now");
  return h("div", {}, h("div", { class: "row" }, h("h2", { class: "m0" }, "Alerts"), check),
    data.alerts.length ? h("table", {}, h("thead", {}, h("tr", {}, ["Rule", "Status", "Severity", "Message", "Since"].map((x) => h("th", {}, x)))),
      h("tbody", {}, data.alerts.map((a) => h("tr", {}, h("td", {}, a.name), h("td", {}, chip(a.status)), h("td", {}, a.status === "firing" ? chip(a.severity) : "-"),
        h("td", {}, a.message, a.details ? h("pre", {}, json(a.details)) : null), h("td", {}, ago(a.since)))))) : notice("No alert has been recorded yet. Press Check now."));
}

const WHAT = { agent: "on = the agent answers tickets; off = tickets go to people", writes: "on = writes follow their modes; frozen = no change is made to any order" };

async function controls() {
  const data = await api("GET", "/v1/controls");
  const out = h("div", {});
  const cards = data.controls.map((c) => {
    const reason = h("input", { placeholder: "reason (optional)", maxlength: "300", class: "w100" });
    return h("div", { class: "card" + " mb12" },
      h("div", { class: "row" + " m0b4" }, h("b", {}, c.name), chip(c.value)),
      h("div", { class: "muted" }, WHAT[c.name] || ""),
      h("div", { class: "muted" + " my4" }, c.updated_at ? "Set by " + c.updated_by + " " + ago(c.updated_at) + (c.reason ? ": " + c.reason : "") : "Default, never changed"),
      reason,
      h("div", { class: "row" + " mt8" }, c.allowed.filter((v) => v !== c.value).map((v) => {
        const stopping = (c.name === "agent" && v === "off") || (c.name === "writes" && v === "frozen");
        const blocked = !stopping && session.role !== "admin";
        const b = h("button", { class: "btn " + (stopping ? "danger" : "primary"), disabled: blocked, title: blocked ? "Only an admin can turn this back on" : null,
          onclick: async () => {
            if (!confirm("Set " + c.name + " to " + v + "?")) return;
            b.disabled = true;
            try { await api("PUT", "/v1/controls/" + c.name, { value: v, reason: reason.value || null }); render(); } catch (e) { b.disabled = false; out.replaceChildren(notice(e.message, true)); }
          } }, "Set " + v);
        return b; })));
  });
  return h("div", {}, h("h2", {}, "Controls"), session.role !== "admin" ? notice("You can stop things here. An admin turns them back on.") : null, out, cards);
}

async function accounts() {
  if (session.role !== "admin") return notice("Admins only.", true);
  const data = await api("GET", "/v1/admin/reviewers");
  const out = h("div", {});
  const act = async (fn) => { try { await fn(); render(); } catch (e) { out.replaceChildren(notice(e.message, true)); } };
  const u = h("input", { placeholder: "username", autocomplete: "off" });
  const p = h("input", { placeholder: "password (12+ characters)", type: "password", autocomplete: "new-password" });
  const r = h("select", {}, h("option", { value: "reviewer" }, "reviewer"), h("option", { value: "admin" }, "admin"));
  return h("div", {}, h("h2", {}, "Accounts"), out,
    h("table", {}, h("thead", {}, h("tr", {}, ["User", "Role", "State", "Last sign-in", ""].map((x) => h("th", {}, x)))),
      h("tbody", {}, data.reviewers.map((v) => h("tr", {}, h("td", {}, v.username), h("td", {}, v.role),
        h("td", {}, v.active ? (v.locked ? chip("locked") : chip("active")) : chip("disabled")), h("td", {}, v.last_login_at ? ago(v.last_login_at) : "never"),
        h("td", {}, h("div", { class: "row" + " m0" },
          h("button", { class: "btn", onclick: () => act(() => api("PUT", "/v1/admin/reviewers/" + v.username + "/active", { active: !v.active })) }, v.active ? "Disable" : "Enable"),
          h("button", { class: "btn", onclick: () => { const np = prompt("New password for " + v.username + " (12+ characters). They will be signed out everywhere."); if (np) act(() => api("PUT", "/v1/admin/reviewers/" + v.username + "/password", { password: np })); } }, "Set password"))))))),
    h("h3", {}, "Add an account"), h("div", { class: "row" }, u, p, r,
      h("button", { class: "btn primary", onclick: () => act(() => api("POST", "/v1/admin/reviewers", { username: u.value, password: p.value, role: r.value })) }, "Create")));
}

const views = { overview, tickets, approvals, dead, alerts, controls, accounts };

// ---------------------------------------------------------------------------- start

(async function start() {
  if (location.hash.slice(1) in views) view = location.hash.slice(1);
  try { session = await api("GET", "/v1/admin/me"); } catch (_) { session = null; }
  render();
})();
