// The 3D view, shared by the workbench (index.html) and the stage (stage.html): the real robot
// description posed from the live joint angles, an orange ghost that plays a plan, a purple
// planned route, a teal path read from a topic, and the cell's keep-out shapes. It talks to a
// page only through window.showLive/visualise/showRoute/clearRoute/showPath/clearPath/hideGhost/
// lookAt and the elements "scene", "viewnote", "view-home" (and "view-sethome", "view-cell" if present).
// The 3D view: the real robot description, posed from the live joint angles.
// The frame names come from config.py through /state, so nothing here names a robot.
const frames = {base: "base", tool: "tool0"};
fetch("/state").then(r => r.json()).then(s => { if (s.frames) Object.assign(frames, s.frames); }).catch(() => {});
const baseLink = () => live.links[frames.base] || live;
// A translucent orange copy ("ghost") plays MoveIt's planned path on request.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import URDFLoader from "urdf-loader";

const el = document.getElementById("scene"), note = document.getElementById("viewnote");
const renderer = new THREE.WebGLRenderer({antialias: true});
renderer.setPixelRatio(window.devicePixelRatio);
el.appendChild(renderer.domElement);
const scene = new THREE.Scene();
scene.background = new THREE.Color(0xffffff);
const camera = new THREE.PerspectiveCamera(38, 1, 0.05, 20);
// Home view: a sensible default, or the one this browser saved with "Set home".
const DEFAULT_HOME = {pos: [1.7, 1.3, 1.7], target: [0, 0.55, 0]};
function savedHome() {
  try { return JSON.parse(localStorage.getItem("robot-console-home")) || DEFAULT_HOME; } catch { return DEFAULT_HOME; }
}
camera.position.fromArray(savedHome().pos);
const controls = new OrbitControls(camera, renderer.domElement);
// Draw only when something changes, so the page leaves the GPU free for the language model.
let dirty = true;
controls.addEventListener("change", () => { dirty = true; });
THREE.DefaultLoadingManager.onProgress = () => { dirty = true; };   // each robot mesh as it arrives
controls.target.fromArray(savedHome().target);
scene.add(new THREE.HemisphereLight(0xffffff, 0x8899aa, 1.2));
const sun = new THREE.DirectionalLight(0xffffff, 0.9); sun.position.set(2, 3, 1); scene.add(sun);
scene.add(new THREE.GridHelper(2, 20, 0xcfd6de, 0xe3e8ee));

let live = null, ghost = null, anim = null, pendingLive = null;
const ghostMaterial = new THREE.MeshStandardMaterial({color: 0xe07b00, transparent: true, opacity: 0.5, depthWrite: false, roughness: 0.6});

function makeLoader(material) {
  const loader = new URDFLoader();
  loader.packages = pkg => "/pkg/" + pkg;   // package://x/... becomes /pkg/x/... on this server
  if (material) {
    // Meshes arrive after the URDF itself, so recolour each one as it lands.
    const original = loader.defaultMeshLoader.bind(loader);
    loader.loadMeshCb = (path, manager, done) => original(path, manager, (mesh, err) => {
      if (mesh) mesh.traverse(o => { if (o.isMesh) o.material = material; });
      done(mesh, err);
    });
  }
  return loader;
}

function load(loader, onDone) {
  loader.load("/static/robot.urdf", robot => {
    robot.rotation.x = -Math.PI / 2;          // ROS is Z-up, three.js is Y-up
    scene.add(robot); dirty = true; onDone(robot);
  }, undefined, err => { note.textContent = "Could not load the robot model: " + err; });
}
let viewReadyResolve;
window.viewReady = new Promise(resolve => { viewReadyResolve = resolve; });
load(makeLoader(null), r => {
  live = r; note.textContent = ""; viewReadyResolve();
  baseLink().add(makeAxes(0.3));      // the frame every x, y, z on this page is measured in
  loadCell();
  if (pendingLive) setPose(live, pendingLive);
});
load(makeLoader(ghostMaterial), r => { ghost = r; ghost.visible = false; });

function setPose(robot, degs) {      // degs is {joint name: degrees}
  if (!robot || !degs) return;
  dirty = true;
  for (const [name, d] of Object.entries(degs)) if (robot.joints[name]) robot.setJointValue(name, d * Math.PI / 180);
}
// Axes in CAD colours (x red, y green, z blue), with letters that always face the camera.
const AXES = [["x", 0xd03b3b, "#d03b3b"], ["y", 0x2e9d4f, "#2e9d4f"], ["z", 0x2f6fdb, "#2f6fdb"]];
function letter(text, css) {
  const c = document.createElement("canvas"); c.width = c.height = 64;
  const g = c.getContext("2d");
  g.font = "600 44px IBM Plex Sans, sans-serif"; g.fillStyle = css; g.textAlign = "center"; g.textBaseline = "middle";
  g.fillText(text, 32, 34);
  return new THREE.Sprite(new THREE.SpriteMaterial({map: new THREE.CanvasTexture(c), depthTest: false}));
}
function makeAxes(length) {
  const group = new THREE.Group();
  const directions = [new THREE.Vector3(1, 0, 0), new THREE.Vector3(0, 1, 0), new THREE.Vector3(0, 0, 1)];
  AXES.forEach(([name, hex, css], i) => {
    group.add(new THREE.ArrowHelper(directions[i], new THREE.Vector3(), length, hex, length * 0.2, length * 0.1));
    const s = letter(name, css);
    s.position.copy(directions[i]).multiplyScalar(length * 1.3);
    s.scale.setScalar(length * 0.5);
    group.add(s);
  });
  return group;
}

// The view cube in the bottom-left corner, like Fusion 360's: it turns with the view, and
// clicking a face looks at the robot from that side. Faces are named after the robot's base
// axes, so "+X" looks along x and shows the y-z plane face-on.
// three.js lays out box faces assuming y is up; the cube is turned a quarter round x below so
// its "up" is the robot's z. The list is in three.js face order, each with its robot direction.
const FACES = [["+X", "#d03b3b", [1, 0, 0]], ["−X", "#d03b3b", [-1, 0, 0]], ["TOP", "#2f6fdb", [0, 0, 1]],
               ["BOTTOM", "#2f6fdb", [0, 0, -1]], ["−Y", "#2e9d4f", [0, -1, 0]], ["+Y", "#2e9d4f", [0, 1, 0]]];
function faceMaterial(text, css) {
  const c = document.createElement("canvas"); c.width = c.height = 128;
  const g = c.getContext("2d");
  g.fillStyle = "#f3f5f8"; g.fillRect(0, 0, 128, 128);
  g.strokeStyle = "#9aa6b2"; g.lineWidth = 6; g.strokeRect(3, 3, 122, 122);
  g.font = "700 40px IBM Plex Sans, sans-serif"; g.fillStyle = css; g.textAlign = "center"; g.textBaseline = "middle";
  if (text === "TOP" || text === "BOTTOM") { g.translate(64, 64); g.rotate(Math.PI); g.translate(-64, -64); }   // read the right way up from the default view
  g.fillText(text, 64, 66);
  return new THREE.MeshBasicMaterial({map: new THREE.CanvasTexture(c)});
}
const cube = new THREE.Mesh(new THREE.BoxGeometry(1.8, 1.8, 1.8), FACES.map(([text, css]) => faceMaterial(text, css)));
cube.rotation.x = Math.PI / 2;                 // three.js y (label up) becomes the robot's z
const key = new THREE.Group();
const keyAxes = makeAxes(2.4);
keyAxes.position.set(-0.9, -0.9, -0.9);    // along the cube's edges from one corner, as in Fusion
key.add(cube, keyAxes);
const keyScene = new THREE.Scene(), keyCamera = new THREE.OrthographicCamera(-2.5, 2.5, 2.5, -2.5, 0.1, 10);
keyScene.add(key);
const KEY_PX = 150, baseTurn = new THREE.Quaternion();
function renderKey() {
  if (live) { baseLink().getWorldQuaternion(baseTurn); key.quaternion.copy(baseTurn); }
  keyCamera.position.copy(camera.position).sub(controls.target).setLength(4);
  keyCamera.up.copy(camera.up); keyCamera.lookAt(0, 0, 0);
  renderer.autoClear = false; renderer.clearDepth();
  renderer.setScissorTest(true); renderer.setScissor(6, 6, KEY_PX, KEY_PX); renderer.setViewport(6, 6, KEY_PX, KEY_PX);
  renderer.render(keyScene, keyCamera);
  renderer.setScissorTest(false); renderer.setViewport(0, 0, el.clientWidth, el.clientHeight);
  renderer.autoClear = true;
}

// Which cube face (if any) is under the mouse.
const raycaster = new THREE.Raycaster();
function faceUnder(e) {
  const r = renderer.domElement.getBoundingClientRect();
  const x = e.clientX - r.left - 6, y = r.bottom - e.clientY - 6;
  if (x < 0 || y < 0 || x > KEY_PX || y > KEY_PX) return null;
  raycaster.setFromCamera(new THREE.Vector2(x / KEY_PX * 2 - 1, y / KEY_PX * 2 - 1), keyCamera);
  const hit = raycaster.intersectObject(cube)[0];
  return hit ? FACES[hit.face.materialIndex] : null;
}
renderer.domElement.addEventListener("pointermove", e => {
  const face = faceUnder(e);
  renderer.domElement.style.cursor = face ? "pointer" : "";
  renderer.domElement.title = face ? `View from ${face[0]}` : "";
});
renderer.domElement.addEventListener("click", e => {
  const face = faceUnder(e);
  if (!face) return;
  const dir = new THREE.Vector3(...face[2]).applyQuaternion(baseTurn);       // base axes to the scene
  if (Math.abs(dir.y) > 0.99) {                                              // straight down or up: keep a heading
    const heading = camera.position.clone().sub(controls.target).setY(0).normalize();
    dir.addScaledVector(heading, 0.002).normalize();
  }
  const distance = camera.position.distanceTo(controls.target);
  flyTo(controls.target.clone().addScaledVector(dir, distance), controls.target.clone());
});

// Swing the camera round to a new view (instantly if the viewer prefers less motion).
let fly = null;
function flyTo(position, target) {
  const t0 = controls.target.clone(), off0 = camera.position.clone().sub(t0), off1 = position.clone().sub(target);
  const turn = new THREE.Quaternion().setFromUnitVectors(off0.clone().normalize(), off1.clone().normalize());
  const T = matchMedia("(prefers-reduced-motion: reduce)").matches ? 0 : 450, start = performance.now();
  fly = () => {
    const k = T ? Math.min(1, (performance.now() - start) / T) : 1, e = k * (2 - k);
    controls.target.lerpVectors(t0, target, e);
    const dir = off0.clone().normalize().applyQuaternion(new THREE.Quaternion().slerp(turn, e));
    camera.position.copy(controls.target).addScaledVector(dir, THREE.MathUtils.lerp(off0.length(), off1.length(), e));
    dirty = true;
    if (k >= 1) fly = null;
  };
}

document.getElementById("view-home").onclick = () => {
  const home = savedHome();
  flyTo(new THREE.Vector3().fromArray(home.pos), new THREE.Vector3().fromArray(home.target));
};
if (document.getElementById("view-sethome")) document.getElementById("view-sethome").onclick = () => {
  try {
    localStorage.setItem("robot-console-home", JSON.stringify({pos: camera.position.toArray(), target: controls.target.toArray()}));
    note.textContent = "This view is now your home view.";
  } catch {
    note.textContent = "This browser would not save the home view.";
  }
  setTimeout(() => { if (note.textContent.startsWith("This")) note.textContent = ""; }, 3000);
};

function resize() {
  const w = el.clientWidth, h = el.clientHeight;
  renderer.setSize(w, h, false); camera.aspect = w / h; camera.updateProjectionMatrix(); dirty = true;
}
new ResizeObserver(resize).observe(el); resize();

window.showLive = degs => { pendingLive = degs; setPose(live, degs); };
window.visualise = (names, path, seconds) => {
  if (!ghost || !path.length) return;
  ghost.visible = true;
  const T = 3000, t0 = performance.now();
  anim = () => {      // step through MoveIt's planned waypoints, sped up to 3 s
    const k = Math.min(1, (performance.now() - t0) / T);
    const point = path[Math.round(k * (path.length - 1))];
    setPose(ghost, Object.fromEntries(names.map((n, i) => [n, point[i]])));
    if (k >= 1) anim = null;
  };
  note.textContent = `Orange plays the planned path, sped up; the real move takes about ${seconds} s.`;
};
// The planned route, drawn as soon as a confirm card appears. The line is where MoveIt will
// really take the tool (worked out from the planned joint angles); the dots are the points
// the model asked for, S being where the tool starts. If the dots are not the shape you
// asked for, the model got it wrong: cancel.
const ROUTE = 0x7c3aed;
let route = null;
window.showRoute = (names, path, pointsCm) => {
  window.clearRoute();
  if (!ghost || !live || !path.length) return;
  route = new THREE.Group();
  const line = [];
  for (const frame of path) {                 // pose the hidden ghost at each step and note where its tool is
    setPose(ghost, Object.fromEntries(names.map((n, i) => [n, frame[i]])));
    ghost.updateMatrixWorld(true);
    line.push(ghost.links[frames.tool].getWorldPosition(new THREE.Vector3()));
  }
  route.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(line),
                           new THREE.LineBasicMaterial({color: ROUTE, depthTest: false})));
  baseLink().updateMatrixWorld(true);
  (pointsCm || []).forEach((p, i) => {
    const colour = i ? ROUTE : 0x1b2430;
    const dot = new THREE.Mesh(new THREE.SphereGeometry(0.007), new THREE.MeshBasicMaterial({color: colour, depthTest: false}));
    dot.position.set(p[0] / 100, p[1] / 100, p[2] / 100).applyMatrix4(baseLink().matrixWorld);   // base frame to the scene
    const tag = letter(i ? String(i) : "S", i ? "#7c3aed" : "#1b2430");
    tag.position.copy(dot.position).add(new THREE.Vector3(0, 0.03, 0));
    tag.scale.setScalar(0.05);
    route.add(dot, tag);
  });
  scene.add(route);
  note.textContent = pointsCm ? "Purple: the planned route. Dots: the points asked for, S is the start." : "Purple: the planned route of the tool.";
  dirty = true;
};
window.clearRoute = () => {
  if (!route) return;
  scene.remove(route); route = null; dirty = true;
  if (!anim && !pathGroup) note.textContent = "";
};

// A path read from one of the system's own topics (for example an inspection raster),
// drawn in teal in the base frame. S is its start, E its end.
const PATH = 0x0f6e78;
let pathGroup = null;
window.showPath = (pointsCm, label, labels = null) => {
  window.clearPath();
  if (!live || !pointsCm || !pointsCm.length) return;
  baseLink().updateMatrixWorld(true);
  const pts = pointsCm.map(p => new THREE.Vector3(p[0] / 100, p[1] / 100, p[2] / 100).applyMatrix4(baseLink().matrixWorld));
  pathGroup = new THREE.Group();
  pathGroup.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts),
                               new THREE.LineBasicMaterial({color: PATH, depthTest: false})));
  // Without labels: S and E at the ends. With labels (one per point, e.g. "1".."24"): a small tag on every point.
  const tagged = labels ? pts.map((p, i) => [p, String(labels[i] ?? i + 1), 0.03]) : [[pts[0], "S", 0.05], [pts[pts.length - 1], "E", 0.05]];
  tagged.forEach(([p, text, size]) => {
    const dot = new THREE.Mesh(new THREE.SphereGeometry(labels ? 0.005 : 0.007), new THREE.MeshBasicMaterial({color: PATH, depthTest: false}));
    dot.position.copy(p);
    const tag = letter(text, "#0f6e78");
    tag.position.copy(p).add(new THREE.Vector3(0, labels ? 0.018 : 0.03, 0));
    tag.scale.setScalar(size);
    pathGroup.add(dot, tag);
  });
  scene.add(pathGroup);
  note.textContent = labels ? `Teal: ${label}, ${pointsCm.length} numbered points.` : `Teal: ${label}, ${pointsCm.length} points (S start, E end).`;
  dirty = true;
};
// Fly the camera to frame some points (in the base frame, centimetres), e.g. a path just shown.
window.lookAt = pointsCm => {
  if (!live || !pointsCm || !pointsCm.length) return;
  baseLink().updateMatrixWorld(true);
  const c = new THREE.Vector3();
  for (const p of pointsCm) c.add(new THREE.Vector3(p[0] / 100, p[1] / 100, p[2] / 100).applyMatrix4(baseLink().matrixWorld));
  c.multiplyScalar(1 / pointsCm.length);
  flyTo(c.clone().add(new THREE.Vector3(0.45, 0.35, 0.45)), c);
};
window.clearPath = () => {
  if (!pathGroup) return;
  scene.remove(pathGroup); pathGroup = null; dirty = true;
  if (!route && !anim) note.textContent = "";
};

// Where the arm may not go: MoveIt's model of the cell (walls, table, keep-out zones),
// drawn as see-through red shapes. MoveIt refuses any plan that touches them.
const cellShapes = [];
async function loadCell() {
  let data;
  try { const r = await fetch("/cell"); if (!r.ok) return; data = await r.json(); } catch { return; }
  const fill = new THREE.MeshBasicMaterial({color: 0xd03b3b, transparent: true, opacity: 0.1, depthWrite: false});
  const edge = new THREE.LineBasicMaterial({color: 0xd03b3b, transparent: true, opacity: 0.45});
  for (const shape of data.shapes) {
    let geometry;
    if (shape.kind === "box") geometry = new THREE.BoxGeometry(...shape.size);
    else if (shape.kind === "sphere") geometry = new THREE.SphereGeometry(shape.size[0], 24, 16);
    else if (shape.kind === "cylinder") geometry = new THREE.CylinderGeometry(shape.size[1], shape.size[1], shape.size[0], 24).rotateX(Math.PI / 2);  // ROS cylinders stand along z
    else continue;
    const solid = new THREE.Mesh(geometry, fill);
    solid.add(new THREE.LineSegments(new THREE.EdgesGeometry(geometry), edge));
    solid.position.fromArray(shape.position);
    solid.quaternion.fromArray(shape.rotation);
    solid.name = shape.name;
    (live.links[shape.frame] || live).add(solid);    // in the frame MoveIt gave it
    cellShapes.push(solid);
  }
  dirty = true;
}
if (document.getElementById("view-cell")) document.getElementById("view-cell").onclick = e => {
  const show = e.target.getAttribute("aria-pressed") !== "true";
  e.target.setAttribute("aria-pressed", String(show));
  e.target.textContent = show ? "Hide cell" : "Show cell";
  cellShapes.forEach(solid => { solid.visible = show; });
  dirty = true;
};

window.hideGhost = () => { if (ghost && ghost.visible) { ghost.visible = false; dirty = true; } anim = null; if (!route) note.textContent = ""; };

(function tick() {
  if (anim) anim();
  if (fly) fly();
  controls.update();
  if (dirty) { renderer.render(scene, camera); renderKey(); dirty = false; }
  requestAnimationFrame(tick);
})();

