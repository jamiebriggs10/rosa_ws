// The settings panels of the stage (stage.html): the requests helper, the robot-connection ladder,
// the voice section (the live part and the form), the safety limits, the session buttons, the model menu, the Ollama address and
// the capabilities panel. Each renderer looks its elements up by id.
const $ = id => document.getElementById(id);
let lastState = null, ollamaFilled = false;

// Called by a page's refresh() with the /state object: every panel the page has is brought up to date.
function panelsRefresh(s) {
  lastState = s;
  if (!ollamaFilled && s.ollama_url && $("ollama-url")) { $("ollama-url").value = s.ollama_url; ollamaFilled = true; }
  if ($("think")) { $("think").checked = s.think; $("think").disabled = s.busy || !modelThinks[s.model]; }
  showLimits(s);
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

// ---- the robot-connection ladder and voice ----
// The robot-connection widget and its dialog. The ladder comes from /state.robot (computed on the
// server every few seconds); the dialog is redrawn only when it changes, so a poll never pulls a
// button out from under a click. Actions post a name; the server decides what it may run.
let robotKey = null, robotMode = "sim", robotLocal = false, robotArmed = null, robotMessage = "";
const MODE_NAMES = {sim: "Simulator", real: "Real robot"};
function showRobot(r, local) {
  robotMode = r.mode || robotMode; robotLocal = local;
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
if ($("robot-mode-sim")) $("robot-mode-sim").onclick = () => robotSetMode("sim");
if ($("robot-mode-real")) $("robot-mode-real").onclick = () => robotSetMode("real");
if ($("robot-recheck")) $("robot-recheck").onclick = () => robotSetMode(robotMode);

// ---- voice ----
// /state.voice is the voice package's latest news merged with the voice block of the settings. The
// live part of the Voice section is redrawn when it changes; the form is filled when the pop-up
// opens (loadVoiceForm), so a value being typed is never overwritten by a poll.
let voiceOn = false, voiceKey = null, voiceMessage = "";
const VOICE_STATES = {idle: ["ok", "idle"], armed: ["wait", "armed, waiting for speech"],
                      listening: ["wait", "listening"], finalising: ["busy", "finishing"]};
const VOICE_ACTIONS = {sent: "sent to the model", stopped: "stopped the arm", confirmed: "confirmed the move",
                       cancelled: "cancelled the move", held: "cancelled; the correction runs next",
                       busy: "not sent: the model was busy", off: "heard, but voice is off", empty: ""};
function voiceLampText(v) {
  if (!v.on) return ["off", "Voice off"];
  if (v.error) return ["off", "Voice on, not working"];
  if (!v.running) return ["wait", "Voice on, " + (v.loading || "starting…")];
  const [lamp, word] = VOICE_STATES[v.state] || ["ok", v.state || "ready"];
  return [lamp, "Voice on, " + word];
}
function showVoice(v) {
  voiceOn = !!v.on;
  const dlg = $("settings");
  if (!dlg || !dlg.open) return;
  const key = JSON.stringify([v, voiceMessage]);
  if (key === voiceKey) return;
  voiceKey = key;
  const [lamp, text] = voiceLampText(v);
  $("voice-on-btn").className = v.on ? "sel" : ""; $("voice-off-btn").className = v.on ? "" : "sel";
  const pill = $("voice-state"); pill.className = "pill " + lamp; pill.textContent = text.replace("Voice on, ", "").replace("Voice ", "");
  const problem = voiceMessage || v.error || "";
  const alert = $("voice-alert");
  alert.hidden = !(problem || v.loading);
  alert.className = "alert " + (problem ? "bad" : "wait"); alert.textContent = problem || v.loading || "";
  $("voice-level").style.width = Math.min(100, Math.round((v.level || 0) * 400)) + "%";
  $("voice-heard").innerHTML = v.final ? esc(v.final) : '<span class="empty-note">Nothing heard yet.</span>';
  $("voice-felt").textContent = v.felt_ms != null ? `felt delay ${Math.round(v.felt_ms)} ms` : "";
  $("voice-action").textContent = v.final ? (VOICE_ACTIONS[v.action] ?? v.action ?? "") : "";
  $("voice-ptt").disabled = !(v.on && v.running);
  $("voice-confirm").checked = !!v.confirm;
}
async function setVoice(on) { voiceKey = null; await post("/voice/on", {on}); refresh(); }
if ($("voice-on-btn")) $("voice-on-btn").onclick = () => setVoice(true);
if ($("voice-off-btn")) $("voice-off-btn").onclick = () => setVoice(false);
if ($("voice-confirm")) $("voice-confirm").onchange = e => post("/voice/confirm", {on: e.target.checked}).then(() => { voiceKey = null; refresh(); });
function voiceProblem(msg) { voiceMessage = msg; voiceKey = null; if (lastState && lastState.voice) showVoice(lastState.voice); }

// The form: one control per key of the voice block, each change posted on its own to /settings/voice.
const VOICE_FORM = [["s-input-device", "input_device", "name"], ["s-output-device", "output_device", "name"],
                    ["s-speech-voice", "speech_voice", "text"], ["s-wake-threshold", "wake_threshold", "number"],
                    ["s-silence-ms", "silence_ms", "int"], ["s-speech-enabled", "speech_enabled", "bool"],
                    ["s-sounds-enabled", "sounds_enabled", "bool"], ["s-verbosity", "verbosity", "int"],
                    ["s-persona", "persona", "text"], ["s-follow", "follow_up_s", "number"],
                    ["s-timeout", "confirm_timeout_s", "number"], ["s-ros", "map_ros", "bool"]];
const VOICE_READ = {name: el => el.value || null, number: el => Number(el.value), int: el => parseInt(el.value, 10),
                    bool: el => el.checked, text: el => el.value};
for (const [id, key, kind] of VOICE_FORM) if ($(id)) $(id).onchange = async e => {
  const r = await fetch("/settings/voice", {method: "POST", headers: {"Content-Type": "application/json"},
                                            body: JSON.stringify({[key]: VOICE_READ[kind](e.target)})});
  const data = await r.json().catch(() => ({}));
  if (!r.ok) { voiceProblem(data.error || "That setting was not saved."); return; }
  voiceMessage = ""; $("voice-saved").textContent = "saved"; setTimeout(() => { $("voice-saved").textContent = ""; }, 2000);
  refresh();
};
function fillSelect(id, names, current, none) {
  const sel = $(id);
  sel.innerHTML = `<option value="">${none}</option>` + names.map(n => `<option value="${esc(n)}">${esc(n)}</option>`).join("");
  if (current && !names.includes(current)) sel.append(Object.assign(document.createElement("option"), {value: current, textContent: current}));
  sel.value = current || "";
}
async function loadVoiceForm() {
  const v = (lastState && lastState.voice) || {};
  let d = {input: [], output: [], voices: []};
  try { d = await (await fetch("/voice/devices")).json(); } catch {}
  fillSelect("s-input-device", d.input || [], v.input_device, "(default microphone)");
  fillSelect("s-output-device", d.output || [], v.output_device, "(default speaker)");
  fillSelect("s-speech-voice", d.voices || [], v.speech_voice, "(no voices in models/piper)");
  for (const [id, key, kind] of VOICE_FORM) {
    if (!$(id) || kind === "name" || key === "speech_voice" || v[key] === undefined) continue;
    if (kind === "bool") $(id).checked = !!v[key]; else $(id).value = v[key];
  }
}
if ($("s-test-voice")) $("s-test-voice").onclick = () =>
  post("/voice/say", {text: "This is Rosa."}).then(r => { if (!r.spoken) voiceProblem("Rosa cannot speak: " + (r.reason || "no reason given")); });

// Hold to talk: down and up go to the voice package. Let go anywhere (leave, blur, close) so the
// microphone is never left open.
let pttHeld = false;
function ptt(down) {
  if (pttHeld === down || (down && $("voice-ptt").disabled)) return;
  pttHeld = down;
  $("voice-ptt").classList.toggle("held", down);
  fetch("/voice/ptt", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({down})}).catch(() => {});
}
const pttBtn = $("voice-ptt");
if (pttBtn) {
  pttBtn.addEventListener("pointerdown", e => { pttBtn.setPointerCapture(e.pointerId); ptt(true); });
  for (const ev of ["pointerup", "pointerleave", "pointercancel"]) pttBtn.addEventListener(ev, () => ptt(false));
  window.addEventListener("blur", () => ptt(false));
  $("settings").addEventListener("close", () => ptt(false));
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

