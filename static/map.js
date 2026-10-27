// The map: what the connected nodes offer, drawn as a graph. Rosa in the middle, one node per
// program (group), one small node per thing it offers, from the rows /capabilities returns.
// Only the described interfaces (and MoveIt's tools) are drawn by default; "show ROS" switches the
// labels to ROS names, adds the undescribed interfaces of groups that have a described one, and
// collapses every other group into one node with a count, so a hundred controller topics never
// bury the fifteen things a person can say. The running command's path and a tour's sentences are
// lit from /state.trace and /state.lit. Layout: d3-force (d3 is loaded by the page before this file).
const RosaMap = (() => {
  const GROUP_NAMES = {moveit: "MoveIt", root: "Root", ur_driver: "UR driver"};
  function groupName(id) { return GROUP_NAMES[id] || String(id).replace(/_/g, " ").replace(/^\w/, c => c.toUpperCase()); }

  // Pure: which groups and offers to draw. Exercised by tests/test_page_ids.py under node.
  function select(rows, showRos) {
    const described = rows.filter(r => r.status !== "undescribed" || r.node === "moveit");
    const ids = [];
    for (const r of described) if (!ids.includes(r.node)) ids.push(r.node);
    const groups = ids.map(id => ({id, label: groupName(id), up: rows.some(r => r.node === id && r.status !== "missing"), collapsed: false}));
    const offers = described.slice();
    if (showRos) {
      const rest = {};
      for (const r of rows) {
        if (r.status !== "undescribed" || r.node === "moveit") continue;
        if (ids.includes(r.node)) offers.push(r); else (rest[r.node] ||= []).push(r);
      }
      for (const [id, list] of Object.entries(rest))
        groups.push({id, label: `${groupName(id)} · ${list.length} more`, up: true, collapsed: true, count: list.length});
    }
    return {groups, offers};
  }

  let svg = null, sim = null, handlers = {}, nodes = [], links = [], layers = {}, lastKey = null;
  const state = {rows: [], showRos: false, lit: [], trace: null};

  function mount(el, h) {
    svg = d3.select(el); handlers = h || {};
    svg.selectAll("*").remove();
    layers.links = svg.append("g"); layers.nodes = svg.append("g");
    new ResizeObserver(() => { if (nodes.length) layout(); }).observe(el);
  }

  function update(rows, opts) {
    Object.assign(state, {rows: rows || []}, opts || {});
    const key = JSON.stringify([state.rows.map(r => [r.name, r.node, r.status, r.moves_robot]), !!state.showRos]);
    if (key !== lastKey) { lastKey = key; build(); }
    applyLit();
  }

  function offerLabel(r) {
    if (state.showRos) return r.type || r.name;
    const say = r.say || r.name;                           // what you would say: short and sayable
    return say.length > 30 ? say.slice(0, 28) + "…" : say;
  }

  function build() {
    const {groups, offers} = select(state.rows, state.showRos);
    const old = new Map(nodes.map(n => [n.id, n]));        // keep positions across redraws
    nodes = [{id: "rosa", kind: "hub"},
             ...groups.map(g => ({...g, kind: "group"})),
             ...offers.map(r => ({id: r.name, kind: "offer", label: offerLabel(r), parent: r.node, moves: !!r.moves_robot,
                                  up: r.status !== "missing", row: r}))];
    for (const n of nodes) { const o = old.get(n.id); if (o) { n.x = o.x; n.y = o.y; } }
    links = [...groups.map(g => ({source: "rosa", target: g.id})),
             ...offers.map(r => ({source: r.node, target: r.name, offer: true}))];
    layout();
  }

  const idOf = x => (x && x.id) || x;

  function layout() {
    if (!svg) return;
    const box = svg.node().getBoundingClientRect(), W = Math.max(300, box.width), H = Math.max(200, box.height);
    const line = layers.links.selectAll("line").data(links, l => idOf(l.source) + ">" + idOf(l.target));
    line.exit().remove(); line.enter().append("line");
    const g = layers.nodes.selectAll("g").data(nodes, n => n.id);
    g.exit().remove();
    const enter = g.enter().append("g");
    const hub = enter.filter(n => n.kind === "hub");
    hub.append("rect").attr("class", "hub").attr("x", -30).attr("y", -15).attr("width", 60).attr("height", 30).attr("rx", 15);
    hub.append("text").attr("class", "hub").attr("text-anchor", "middle").attr("dy", 4).text("Rosa");
    const rest = enter.filter(n => n.kind !== "hub");
    rest.append("circle"); rest.append("text").attr("text-anchor", "middle"); rest.append("title");
    const all = enter.merge(g);
    all.select("circle").attr("r", n => n.kind === "group" ? 20 : 11)
      .attr("class", n => [n.kind, n.moves ? "moves" : "", n.up ? "" : "off", n.collapsed ? "collapsed" : ""].join(" ").trim())
      .on("click", (e, n) => {
        if (n.kind === "group" && !n.collapsed && handlers.onNode) handlers.onNode(n.id, n.label);
        else if (n.kind === "offer" && handlers.onOffer) handlers.onOffer(n.row);
      });
    all.filter(n => n.kind !== "hub").select("text").attr("class", n => n.kind === "group" ? "" : "s").attr("dy", n => n.kind === "group" ? 32 : 22).text(n => n.label);
    all.select("title").text(n => n.kind === "group" ? (n.collapsed ? `${n.count} undescribed interfaces (read-only)` : `Tap: what can I do with ${n.label}`)
                                                     : `Tap to say: ${n.row ? n.row.say : ""}`);
    if (sim) sim.stop();
    sim = d3.forceSimulation(nodes)
      .force("link", d3.forceLink(links).id(n => n.id).distance(l => l.offer ? 62 : Math.min(W, H) * 0.32).strength(l => l.offer ? 1 : 0.6))
      .force("charge", d3.forceManyBody().strength(n => n.kind === "hub" ? -500 : n.kind === "group" ? -300 : -130))
      .force("center", d3.forceCenter(W / 2, H / 2))
      .force("collide", d3.forceCollide(n => n.kind === "group" ? 40 : 30))
      .on("tick", () => {
        for (const n of nodes) { n.x = Math.max(30, Math.min(W - 30, n.x)); n.y = Math.max(22, Math.min(H - 26, n.y)); }
        layers.links.selectAll("line").attr("x1", l => l.source.x).attr("y1", l => l.source.y).attr("x2", l => l.target.x).attr("y2", l => l.target.y);
        layers.nodes.selectAll("g").attr("transform", n => `translate(${n.x},${n.y})`);
      });
    const centre = nodes.find(n => n.id === "rosa");
    centre.fx = W / 2; centre.fy = H / 2;
    applyLit();
  }

  function applyLit() {
    if (!svg) return;
    const lit = new Set(state.lit || []);
    if (state.trace && state.trace.tool) lit.add(state.trace.tool);
    for (const id of [...lit]) { const n = nodes.find(x => x.id === id); if (n && n.parent) lit.add(n.parent); }
    layers.nodes.selectAll("circle").classed("lit", n => lit.has(n.id));
    layers.links.selectAll("line").classed("lit", l => lit.has(idOf(l.target)) && (lit.has(idOf(l.source)) || idOf(l.source) === "rosa"));
  }

  return {mount, update, select, groupName};
})();
