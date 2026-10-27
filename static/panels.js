// The settings panels, shared by the workbench (index.html) and the stage (stage.html): the
// requests helper, the saved moves, the robot-connection widget and ladder, the voice switch and
// the voice app's settings, the tool buttons, the safety limits, the session buttons, the model
// menu, the Ollama address and the capabilities panel. Each renderer looks its elements up by id
// and does nothing on a page that lacks them, so both pages load this one file.
const $ = id => document.getElementById(id);
let lastState = null, ollamaFilled = false;

// Called by a page's refresh() with the /state object: every panel the page has is brought up to date.
function panelsRefresh(s) {
  lastState = s;
  if (!ollamaFilled && s.ollama_url && $("ollama-url")) { $("ollama-url").value = s.ollama_url; ollamaFilled = true; }
  if ($("think")) { $("think").checked = s.think; $("think").disabled = s.busy || !modelThinks[s.model]; }
  showLimits(s);
  showSaved(s.saved || []);
  showTools(s.tools || [], !!s.pendant);
  if (s.robot) showRobot(s.robot, !!s.local);
  if (s.voice) showVoice(s.voice);
}

// ---- requests and small helpers ----
async function post(url, body) {
  const r = await fetch(url, {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body || {})});
  const data = await r.json();
  if (!r.ok) alert(data.error || "Something went wrong");
  return data;
}

function esc(s) { return String(s).replace(/[&<>]/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;"}[c])); }

// ---- the saved moves list ----
// The saved moves list, kept in a file on the robot computer so it survives restarts.
let savedShown = "";
function showSaved(saved) {
  if (!$("saved-list")) return;
  const key = JSON.stringify(saved);
  if (key === savedShown) return;
  savedShown = key;
  $("saved-count").textContent = saved.length;
  $("saved-list").innerHTML = saved.length ? "" : '<li><span class="what">Nothing saved yet. Press Save under a move in the chat.</span></li>';
  saved.forEach((move, i) => {
    const li = document.createElement("li");
    li.innerHTML = `<span class="what">${esc(move.name)} <span class="mono">${esc(move.tool)}</span> <span>${esc(move.model || "")}, ${esc(move.saved || "")}</span></span>`;
    const here = document.createElement("button"); here.textContent = "Run here";
    here.title = "Do the same move starting from where the tool is now";
    here.onclick = () => post("/rerun", move).then(refresh);
    li.appendChild(here);
    if (move.start_cm) {
      const start = document.createElement("button"); start.textContent = "From its start";
      start.title = `Go back to where it first started (${move.start_cm.join(", ")} cm), then do it`;
      start.onclick = () => post("/rerun", {...move, from_start: true}).then(refresh);
      li.appendChild(start);
    }
    const del = document.createElement("button"); del.textContent = "Delete";
    del.onclick = () => post("/unsave", {index: i}).then(refresh);
    li.appendChild(del);
    $("saved-list").appendChild(li);
  });
}


// ---- the robot-connection widget and dialog, voice, the tool buttons ----
// The robot-connection widget and its dialog. The ladder comes from /state.robot (computed on the
// server every few seconds); the dialog is redrawn only when it changes, so a poll never pulls a
// button out from under a click. Actions post a name; the server decides what it may run.
let robotKey = null, robotMode = "sim", robotLocal = false, robotArmed = null, robotMessage = "";
const MODE_NAMES = {sim: "Simulator", real: "Real robot"};
function showRobot(r, local) {
  robotMode = r.mode || robotMode; robotLocal = local;
  if ($("robot-lamp")) {
    $("robot-lamp").className = "lamp " + (r.overall ? r.overall.lamp : "");
    $("robot-text").textContent = `${MODE_NAMES[robotMode] || "Robot"}: ${r.overall ? r.overall.text : "…"}`;
  }
  if (!$("robot-ladder")) return;
  const key = JSON.stringify([r, local, robotArmed, robotMessage]);
  if (key === robotKey) return;
  robotKey = key;
  for (const m of ["sim", "real"]) $("robot-mode-" + m).className = m === robotMode ? "sel" : "";
  const overall = $("robot-overall"); overall.className = "pill " + (r.overall ? r.overall.lamp : ""); overall.textContent = r.overall ? r.overall.text : "…";
  const alert = $("robot-alert");
  if (robotMessage) { alert.className = "alert bad"; alert.textContent = robotMessage; }
  else { alert.className = "alert " + (r.alert ? r.alert.lamp : ""); alert.textContent = r.alert ? r.alert.text : ""; }
  const ol = $("robot-ladder"); ol.innerHTML = "";
  (r.rungs || []).forEach((rung, i) => {
    const li = document.createElement("li"); li.className = "step";
    const act = document.createElement("div"); act.className = "act";
    if (rung.action) {
      const b = document.createElement("button"); b.type = "button";
      const needsLocal = robotMode === "real" && !local;
      const armed = robotArmed === rung.action.name;
      b.textContent = armed ? "Press again to confirm" : rung.action.label;
      if (rung.action.name === "driver_start") b.classList.add("primary");
      if (needsLocal) { b.disabled = true; b.title = "Only from the robot's computer's own screen"; }
      b.onclick = () => robotAction(rung.action.name);
      act.appendChild(b);
    }
    if (rung.yours) { const y = document.createElement("span"); y.className = "you"; y.textContent = "you do this"; act.appendChild(y); }
    const text = document.createElement("div");
    text.innerHTML = `<div class="t"></div><div class="d"></div>`;
    text.firstChild.textContent = `${i + 1}. ${rung.title}`; text.lastChild.textContent = rung.detail || "";
    const lamp = document.createElement("span"); lamp.className = "lamp " + rung.lamp;
    li.append(lamp, text, act); ol.appendChild(li);
  });
  $("robot-note").textContent = robotMode === "real"
    ? "Nothing here moves the arm. Starting the driver, powering on and releasing the brakes only work from this computer's own screen; Play and the speed slider stay on the pendant."
    : "The simulator's checks. Switch to Real robot when the cable is in.";
}
async function robotAction(name) {
  // Powering on or releasing the brakes of the real arm takes two presses.
  if (robotMode === "real" && (name === "power_on" || name === "brake_release") && robotArmed !== name) {
    robotArmed = name; robotKey = null; refresh(); return;
  }
  robotArmed = null; robotMessage = "";
  let r;
  try { r = await fetch("/robot/action", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({name})}); } catch { return; }
  const data = await r.json().catch(() => ({}));
  if (!r.ok) robotMessage = data.error || "Something went wrong";
  robotKey = null; refresh();
}
async function robotSetMode(mode) {
  robotArmed = null; robotMessage = "";
  await fetch("/robot/mode", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({mode})}).catch(() => {});
  robotKey = null; refresh();
}
if ($("robot-widget")) $("robot-widget").onclick = () => { robotMessage = ""; $("robot-dialog").showModal(); };
if ($("robot-mode-sim")) $("robot-mode-sim").onclick = () => robotSetMode("sim");
if ($("robot-mode-real")) $("robot-mode-real").onclick = () => robotSetMode("real");
if ($("robot-recheck")) $("robot-recheck").onclick = () => robotSetMode(robotMode);

// Voice control. The state comes from /state.voice once a second (the header pill, and the live
// part of the dialog when it is open). The settings are fetched from the voice app, through the
// console's /voice/* routes, when the dialog opens, and redrawn only then and after an option
// changes, so a poll never pulls a control out from under a click.
let voiceOn = false, voiceKey = null, voiceUrlFilled = false, voiceSettings = null, voiceOptions = null;
let voiceMessage = "";       // the voice app's last refusal (a 400 on a setting, say); shown until the next success
let voiceHelp = {stages: {}, options: {}, knobs: {}};         // plain-English help from the voice app's help.json
const VOICE_STAGES = ["trigger", "vad", "endpoint", "stt"];
const VOICE_STAGE_TITLES = {trigger: "when to start listening", vad: "is someone speaking?",
                            endpoint: "have you finished?", stt: "speech-to-text"};
const VOICE_STATES = {idle: ["ok", "idle"], armed: ["wait", "armed, waiting for speech"],
                      listening: ["wait", "listening"], finalising: ["busy", "finishing"]};
const VOICE_ACTIONS = {sent: "sent to the model", stopped: "stopped the arm", swallowed: "already stopped",
                       confirmed: "confirmed the move", cancelled: "cancelled the move",
                       busy: "not sent: the model was busy", off: "heard, but voice is off", empty: "", ignored: ""};
function voiceLampText(v) {
  if (!v.on) return ["off", "Voice off"];
  if (!v.running) return ["off", "Voice on, app not running"];
  const [lamp, word] = VOICE_STATES[v.state] || ["ok", v.state || "ready"];
  return [lamp, "Voice on, " + word];
}
function showVoice(v) {
  voiceOn = !!v.on;
  const [lamp, text] = voiceLampText(v);
  if ($("voice-lamp")) { $("voice-lamp").className = "lamp " + lamp; $("voice-text").textContent = text; }
  if (!voiceUrlFilled && v.url && $("voice-url")) { $("voice-url").value = v.url; voiceUrlFilled = true; }
  const dlg = $("voice-dialog") || $("settings");         // the workbench's pop-up, or the stage's settings
  if (!dlg || !dlg.open) return;
  const key = JSON.stringify([v, voiceMessage]);
  if (key === voiceKey) return;
  voiceKey = key;
  $("voice-on-btn").className = v.on ? "sel" : ""; $("voice-off-btn").className = v.on ? "" : "sel";
  const pill = $("voice-state"); pill.className = "pill " + lamp; pill.textContent = text.replace("Voice on, ", "").replace("Voice ", "");
  const alert = $("voice-alert");
  const problem = voiceMessage || (!v.running
    ? `The voice app is not answering at ${v.url}. Start it (start.sh with "voice_start" set, or its app.py); this page finds it by itself.`
    : (v.error || ""));
  alert.hidden = !(problem || v.loading);
  alert.className = "alert " + (problem ? "bad" : "wait"); alert.textContent = problem || v.loading || "";
  $("voice-level").style.width = Math.min(100, Math.round((v.level || 0) * 400)) + "%";
  const heard = $("voice-heard");
  if (v.partial) heard.innerHTML = `<span class="partial">${esc(v.partial)}…</span>`;
  else if (v.final) heard.textContent = v.final;
  else heard.innerHTML = '<span class="empty-note">Nothing heard yet.</span>';
  $("voice-felt").textContent = v.felt_ms != null && !v.partial ? `felt delay ${Math.round(v.felt_ms)} ms` : "";
  $("voice-action").textContent = v.final && !v.partial ? (VOICE_ACTIONS[v.action] ?? v.action ?? "") : "";
  $("voice-ptt").disabled = !(v.on && v.running);
  $("voice-confirm").checked = !!v.confirm;
  for (const stage of VOICE_STAGES) {                        // "loading…" on the stage whose model is loading
    const badge = $("voice-load-" + stage);
    if (badge) badge.style.display = v.loading && v.loading.includes(stage) ? "" : "none";
  }
}
async function setVoice(on) {
  voiceKey = null;
  await post("/voice/on", {on});                             // the note (app not answering) shows in the dialog's alert
  refresh();
}
if ($("voice-toggle")) $("voice-toggle").onclick = () => setVoice(!voiceOn);
if ($("voice-on-btn")) $("voice-on-btn").onclick = () => setVoice(true);
if ($("voice-off-btn")) $("voice-off-btn").onclick = () => setVoice(false);
if ($("voice-settings")) $("voice-settings").onclick = () => {
  voiceKey = null; voiceMessage = "";
  $("voice-dialog").showModal();
  if (lastState && lastState.voice) showVoice(lastState.voice);
  loadVoiceSettings();
};

// mk("div", {className: "x", on: {click: fn}}, child, ...) builds an element; text only, so no HTML injection.
function mk(tag, attrs, ...kids) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (k === "on") for (const [ev, fn] of Object.entries(v)) node.addEventListener(ev, fn);
    else if (k in node && k !== "list") node[k] = v; else node.setAttribute(k, v);
  }
  for (const kid of kids) if (kid !== "" && kid != null) node.append(kid);
  return node;
}
async function vapi(path, method, body) {                    // one request to the voice app, via the console
  const r = await fetch("/voice/" + path, {method: method || "GET", headers: body ? {"Content-Type": "application/json"} : {},
                                            body: body ? JSON.stringify(body) : undefined});
  const data = await r.json().catch(() => ({}));
  if (!r.ok) { const e = new Error(data.error || r.statusText); e.status = r.status; throw e; }
  return data;
}
// A refusal from the voice app stays on screen (through the one-second redraws) until the next success.
function voiceProblem(msg) { voiceMessage = msg; voiceKey = null; if (lastState && lastState.voice) showVoice(lastState.voice); }
function voiceFine() { if (voiceMessage) { voiceMessage = ""; voiceKey = null; } }
async function loadVoiceSettings() {
  try {
    const [o, s, help, presets] = await Promise.all([vapi("options"), vapi("settings"), vapi("help").catch(() => ({})),
                                                     vapi("presets").catch(() => [])]);
    voiceFine();
    voiceOptions = o.options; voiceSettings = s;
    voiceHelp = {stages: help.stages || {}, options: help.options || {}, knobs: help.knobs || {}};
    renderVoiceStages();
    const sel = $("voice-preset"); sel.innerHTML = "";
    for (const p of presets) sel.append(mk("option", {value: p}, p));
    $("voice-settings-body").hidden = false;
  } catch (e) {
    $("voice-settings-body").hidden = true;
    voiceProblem("Cannot read the voice settings: " + e.message);
  }
}
// Changes are gathered for 400 ms (so a slider or typing sends once), then posted as one patch.
let voicePatch = {}, voiceTimer = null;
function voiceChange(path, value) {                          // path like ["stt", "parakeet", "model"]
  let node = voicePatch;
  path.slice(0, -1).forEach(k => { node = node[k] ||= {}; });
  node[path[path.length - 1]] = value;
  clearTimeout(voiceTimer);
  voiceTimer = setTimeout(voiceFlush, 400);
}
async function voiceFlush() {
  const body = voicePatch; voicePatch = {};
  try {
    voiceSettings = await vapi("settings", "POST", body);
    voiceFine();
    $("voice-saved").textContent = "saved, applies to the next thing you say";
    setTimeout(() => { $("voice-saved").textContent = ""; }, 3000);
  } catch (e) { voiceProblem("Voice settings: " + e.message); loadVoiceSettings(); }
}
function knobInput(path, spec, value) {
  const send = v => voiceChange(path, v);
  const cur = value === undefined ? spec.default : value;
  switch (spec.type) {
    case "bool": return mk("input", {type: "checkbox", checked: !!cur, on: {change: e => send(e.target.checked)}});
    case "choice": return mk("select", {on: {change: e => send(e.target.value)}},
      ...(spec.choices || []).map(c => mk("option", {value: c, selected: c === cur}, String(c))));
    case "int": case "float": {
      const isInt = spec.type === "int";
      return mk("input", {type: "number", value: cur ?? "", min: spec.min ?? "", max: spec.max ?? "", step: spec.step ?? (isInt ? 1 : "any"),
        on: {change: e => { const n = isInt ? parseInt(e.target.value, 10) : parseFloat(e.target.value); if (!isNaN(n)) send(n); }}});
    }
    case "list": return mk("input", {type: "text", value: (cur || []).join(", "),
      on: {change: e => send(e.target.value.split(",").map(s => s.trim()).filter(Boolean))}});
    default: return mk("input", {type: "text", value: cur ?? "", on: {change: e => send(e.target.value)}});
  }
}
function explain(entry, label) {                             // a folded "What is this?" from one help.json entry
  if (!entry || !entry.what) return "";
  const body = mk("div", {}, mk("p", {}, entry.what));
  if (entry.higher) body.append(mk("p", {}, mk("b", {}, "Turn it up: "), entry.higher));
  if (entry.lower) body.append(mk("p", {}, mk("b", {}, "Turn it down: "), entry.lower));
  if (entry.start) body.append(mk("p", {}, mk("b", {}, "Start with: "), entry.start));
  return mk("details", {className: "explain"}, mk("summary", {}, label || "What is this?"), body);
}
function knobRow(path, name, spec, value) {
  return mk("div", {className: "knob"}, mk("label", {}, name), knobInput(path, spec, value),
    spec.help ? mk("div", {className: "help"}, spec.help) : "", explain(voiceHelp.knobs[path.join(".")]));
}
const STT_COMMON = {                                         // knobs every speech-to-text engine shares; /options does not list them
  mode: {type: "choice", choices: ["batch", "streaming"], help: "batch = turn the words into text after you finish; streaming = live text while you speak"},
  partial_every_ms: {type: "int", min: 100, max: 5000, step: 100, help: "how often live text is produced"},
  language: {type: "str"},
  prompt: {type: "str", help: "words the engine should expect (the robot's name, units, directions)"},
};
function renderVoiceStages() {
  const root = $("voice-stages");
  root.replaceChildren();
  $("voice-device").value = voiceSettings.device || "auto";
  for (const stage of VOICE_STAGES) {
    const opts = voiceOptions[stage] || {}, cfg = voiceSettings[stage] || {};
    const sel = mk("select", {className: "use", on: {change: e => {
      cfg.use = e.target.value; voiceChange([stage, "use"], cfg.use); renderVoiceStages(); }}});
    for (const [name, o] of Object.entries(opts)) {
      const op = mk("option", {value: name, selected: name === cfg.use}, name + (o.supported ? "" : " (unavailable here)"));
      if (!o.supported) { op.disabled = true; op.title = o.reason || "not supported on this computer"; }
      sel.append(op);
    }
    const box = mk("div", {className: "stage"},
      mk("h4", {}, stage, mk("small", {}, VOICE_STAGE_TITLES[stage] || ""),
        mk("span", {id: "voice-load-" + stage, className: "badge", style: "display:none"}, "loading…")),
      explain(voiceHelp.stages[stage], "What is this?"), sel,
      explain(voiceHelp.options[`${stage}.${cfg.use}`], "About this option"));
    if (stage === "stt")
      for (const [k, spec] of Object.entries(STT_COMMON)) box.append(knobRow([stage, k], k, spec, cfg[k]));
    // "both" (wake word or push-to-talk) uses the wake word's knobs, kept under trigger.openwakeword.
    const owner = (stage === "trigger" && cfg.use === "both") ? "openwakeword" : cfg.use;
    const o = opts[owner];
    if (o) for (const [k, spec] of Object.entries(o.knobs || {})) box.append(knobRow([stage, owner, k], k, spec, (cfg[owner] || {})[k]));
    root.append(box);
  }
}
if ($("voice-device")) $("voice-device").onchange = e => voiceChange(["device"], e.target.value);
if ($("voice-preset-load")) $("voice-preset-load").onclick = async () => {
  const name = $("voice-preset").value;
  if (!name) return;
  try { voiceSettings = await vapi(`presets/${name}/load`, "POST", {}); renderVoiceStages(); $("voice-saved").textContent = `preset ${name} loaded`; }
  catch (e) { voiceProblem("Preset: " + e.message); }
};
if ($("voice-preset-save")) $("voice-preset-save").onclick = async () => {
  // Presets used for reported results must not change by accident: replacing one takes a second press.
  const btn = $("voice-preset-save"), name = $("voice-preset-name").value.trim();
  if (!name) return;
  try {
    await vapi(`presets/${name}` + (btn.dataset.confirm === name ? "?overwrite=1" : ""), "POST", {});
  } catch (e) {
    if (e.status === 409 && btn.dataset.confirm !== name) { btn.dataset.confirm = name; btn.textContent = "Press again to replace it"; voiceProblem(e.message); return; }
    voiceProblem("Preset: " + e.message); return;
  }
  delete btn.dataset.confirm; btn.textContent = "Save as preset";
  $("voice-saved").textContent = `preset ${name} saved`;
  loadVoiceSettings();
};
// Hold to talk: down and up go to the voice app. Let go anywhere (leave, blur, close) so the
// microphone is never left open.
let pttHeld = false;
function ptt(down) {
  if (pttHeld === down || (down && $("voice-ptt").disabled)) return;
  pttHeld = down;
  $("voice-ptt").classList.toggle("held", down);
  fetch(`/voice/ptt/${down ? "down" : "up"}`, {method: "POST"}).catch(() => {});
}
const pttBtn = $("voice-ptt");
if (pttBtn) {
  pttBtn.addEventListener("pointerdown", e => { pttBtn.setPointerCapture(e.pointerId); ptt(true); });
  for (const ev of ["pointerup", "pointerleave", "pointercancel"]) pttBtn.addEventListener(ev, () => ptt(false));
  window.addEventListener("blur", () => ptt(false));
  if ($("voice-dialog")) $("voice-dialog").addEventListener("close", () => ptt(false));
}
if ($("voice-url-form")) $("voice-url-form").onsubmit = async e => {
  e.preventDefault();
  const note = $("voice-url-note"); note.textContent = "testing…";
  const r = await fetch("/voice/url", {method: "POST", headers: {"Content-Type": "application/json"},
                                       body: JSON.stringify({url: $("voice-url").value.trim()})});
  const data = await r.json().catch(() => ({}));
  note.textContent = r.ok ? `saved: ${data.url}` : (data.error || "Something went wrong");
  if (r.ok) { voiceKey = null; setTimeout(loadVoiceSettings, 1500); }
};
if ($("voice-confirm")) $("voice-confirm").onchange = e => post("/voice/confirm", {on: e.target.checked}).then(() => { voiceKey = null; refresh(); });

let toolsKey = null, toolHint = "", toolHintUntil = 0;
function showTools(tools, pendant) {
  const usable = tools.filter(t => t.available);
  if ($("tools")) $("tools").hidden = usable.length === 0 && !pendant;
  if (!$("tools-buttons")) return;
  const key = JSON.stringify(tools);
  if (key === toolsKey) return;
  toolsKey = key;
  const box = $("tools-buttons"); box.innerHTML = "";
  for (const t of usable) {
    const b = document.createElement("button"); b.type = "button"; b.dataset.tool = t.name;
    const stops = t.name === "record" && t.running;
    b.textContent = stops ? "Stop recording" : t.label;
    if (t.running) { b.classList.add("on"); b.title = "running"; }
    b.onclick = () => pressTool(t.name, stops);
    box.appendChild(b);
  }
}
async function pressTool(name, stop) {
  let r;
  try { r = await fetch(`/tools/${name}` + (stop ? "/stop" : ""), {method: "POST"}); } catch { return; }
  if (!r.ok) {
    const data = await r.json().catch(() => ({}));
    toolHint = data.error || "Something went wrong"; toolHintUntil = Date.now() + 4000;
    if ($("hint")) { $("hint").hidden = false; $("hint").textContent = toolHint; }
  }
  refresh();
}


// ---- the safety limits ----
// The safety limits bar: what the safety layer currently allows, and a form to change it.
function showLimits(s) {
  const L = s.limits;
  if (!$("limits-chips")) return;
  $("limits-chips").innerHTML = [
    `One move <b>≤ ${L.MAX_STEP_CM} cm</b>`, `Path within <b>${L.MAX_PATH_REACH_CM} cm</b> of its start`,
    `Joint turn <b>≤ ${L.MAX_JOINT_DEG}°</b>`, `Path <b>≤ ${L.MAX_PATH_STEPS}</b> points`, `Speed <b>${Math.round(s.speed * 100)}%</b>`,
  ].map(t => `<span class="chip">${t}</span>`).join(" ");
}
if ($("limits-edit")) $("limits-edit").onclick = () => {
  const s = lastState, form = $("limits-form");
  if (!s) return;
  form.innerHTML = Object.keys(s.limits).map(name =>
    `<label>${esc(s.labels[name])} <input type="number" name="${name}" value="${s.limits[name]}" min="1" max="${s.ceilings[name]}" step="any" required></label>`).join("") +
    '<button type="submit">Save</button><button type="button" id="limits-cancel">Cancel</button>' +
    '<span class="note">Each has a ceiling set in config.py that cannot be exceeded from here. Speed is fixed. Every change is noted in the chat and the session table.</span>';
  form.hidden = false;
  $("limits-cancel").onclick = () => { form.hidden = true; };
};
if ($("limits-form")) $("limits-form").onsubmit = async e => {
  e.preventDefault();
  const changes = {};
  for (const input of e.target.querySelectorAll("input")) changes[input.name] = Number(input.value);
  const r = await post("/limits", changes);
  if (r.ok) { e.target.hidden = true; refresh(); }
};

// ---- the session: new conversation, finish ----
if ($("reset")) $("reset").onclick = () => post("/reset").then(refresh);
if ($("think")) $("think").onchange = e => post("/think", {on: e.target.checked}).then(refresh);
if ($("finish")) $("finish").onclick = async () => {
  const r = await post("/finish");
  const rows = r.records.map(x => {
    const calls = x.events.map(e => `${e.tool} ${Object.keys(e.args).length ? JSON.stringify(e.args) : ""} (${e.outcome})`).join("; ") || "none";
    const motion = x.events.filter(e => e.to_motion_s !== undefined).map(e => e.to_motion_s + " s").join(", ");
    const t = x.timings || {};
    return `<tr><td>${x.time}</td><td>${esc(x.command)}</td><td>${esc(x.via || "typed")}</td><td>${t.felt || ""}</td><td>${t.think || ""}</td><td>${t.to_confirm || ""}</td><td>${t.motion || ""}</td><td class="mono">${esc(calls)}</td><td>${motion}</td><td>${x.total_s} s</td></tr>`;
  }).join("");
  $("summary").innerHTML = `<h2 style="margin:0 0 4px;font-size:17px">Session summary</h2>
    <p>${r.records.length} command${r.records.length === 1 ? "" : "s"}. Saved to <span class="mono">${esc(r.path)}</span></p>
    <table><tr><th>Time</th><th>Command</th><th>Via</th><th>Felt</th><th>Think</th><th>To confirm</th><th>Motion</th><th>Tool calls</th><th>To motion</th><th>Total</th></tr>${rows}</table>
    <button onclick="this.closest('dialog').close()">Close</button>`;
  $("summary").showModal();
};


// ---- the model menu, the Ollama address, what the system offers ----
// The model menu: every model Ollama has; ones that cannot call tools are shown but disabled.
const modelThinks = {};
async function loadModels() {
  if (!$("model")) return;
  const r = await fetch("/models"), data = await r.json();
  if (!r.ok) { $("model").innerHTML = `<option>${esc(data.error)}</option>`; return; }
  data.models.forEach(m => { modelThinks[m.name] = m.thinks; });
  $("model").innerHTML = data.models.map(m =>
    `<option value="${esc(m.name)}" ${m.tools ? "" : "disabled"} ${m.name === data.current ? "selected" : ""}>${esc(m.name)}${m.tools ? (m.thinks ? " (can think)" : "") : " (no tool calling)"}</option>`).join("");
}
if ($("model")) $("model").onchange = async e => {
  const r = await post("/model", {name: e.target.value});
  if (!r.ok) loadModels();          // put the menu back if the switch was refused
  refresh();
};
loadModels();

// Settings: another Ollama. Tested before it is saved, and the error shown here, not in a dialog.
if ($("ollama-form")) $("ollama-form").onsubmit = async e => {
  e.preventDefault();
  const note = $("ollama-note");
  note.textContent = "Testing…";
  const r = await fetch("/ollama", {method: "POST", headers: {"Content-Type": "application/json"},
                                    body: JSON.stringify({url: $("ollama-url").value.trim()})});
  const d = await r.json();
  note.textContent = r.ok ? `Saved. Models there: ${d.models.join(", ")}` : d.error;
  if (r.ok) { loadModels(); refresh(); }
};

// What this system offers: the discovered interfaces with their meanings.
async function loadCapabilities() {
  let r, d;
  if (!$("caps-body")) return;
  try { r = await fetch("/capabilities"); d = await r.json(); } catch { return; }
  if (!r.ok) { $("caps-body").textContent = d.error; $("caps-count").textContent = "?"; return; }
  $("caps-system").textContent = d.system || "No description file yet, so everything below is read-only. See the README for how to write one.";
  $("caps-file").textContent = d.file;
  $("caps-count").textContent = d.interfaces.length;
  const order = {service: 0, topic: 1, action: 2, "": 3};
  const rows = [...d.interfaces].sort((a, b) => (a.status === "ready" ? 0 : 1) - (b.status === "ready" ? 0 : 1) || order[a.kind] - order[b.kind] || a.name.localeCompare(b.name));
  $("caps-body").innerHTML = `<table><tr><th>Interface</th><th>Means</th><th></th></tr>` + rows.map(i =>
    `<tr><td><span class="mono">${esc(i.name)}</span><br><span class="type">${esc(i.kind)} ${esc(i.type)}</span></td>
         <td>${esc(i.means)}${i.moves_robot ? ' <span class="moves">moves robot</span>' : ""}</td>
         <td><span class="st ${i.status}">${i.status === "undescribed" ? "read-only" : i.status}</span></td></tr>`).join("") + "</table>";
}
if ($("caps-rescan")) $("caps-rescan").onclick = async () => {
  const r = await fetch("/capabilities/rescan", {method: "POST", headers: {"Content-Type": "application/json"}, body: "{}"});
  if (!r.ok) $("caps-body").textContent = (await r.json()).error; else loadCapabilities();
};
loadCapabilities();

