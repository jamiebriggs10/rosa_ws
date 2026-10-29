# Lichtblick: the engineer's view

Lichtblick ([lichtblick-suite/lichtblick](https://github.com/lichtblick-suite/lichtblick), MPL-2.0)
is the open fork of Foxglove Studio: a browser app that shows every topic, frame, message,
image and plot of a running ROS 2 system. The console page shows it in the World panel behind
the **Lichtblick** tab, next to Rosa's own view, which stays the operator's.

Two pieces, both started by `start.sh sim|real`:

- **foxglove_bridge** (`ros-humble-foxglove-bridge`, MIT), a ROS node in the robot session that
  serves topics, parameters and the robot meshes over one WebSocket, port 8765. The example
  bringup scripts launch it when it is installed.
- **Lichtblick's web build**, the official Docker image plus `default-layout.json` here
  (`Dockerfile`, built by `start.sh` in seconds), on port 8081. The layout: one 3D panel, the
  robot from `/robot_description`, camera 3 m out, labels off, so the tab shows the robot with
  no clicks.

The page builds the address itself: the `lichtblick_url` setting (default
`http://localhost:8081`, "localhost" read as the console's host) plus
`?ds=foxglove-websocket&ds.url=ws://<console host>:8765`. The iframe exists only while the tab
is chosen, because Lichtblick renders continuously and the browser shares the GPU with Ollama.

To change the layout: arrange it in the full app (`http://localhost:8081`), export it from the
Layouts menu, and replace `default-layout.json`; the next `start.sh` rebuilds the image with it.
Chrome gives an iframe its own storage, so a layout arranged inside the tab is not kept.
