# Lichtblick's official web build with our layout baked in. start.sh builds this (seconds,
# cached) and runs it on port 8081. A build is used rather than a bind mount because Docker
# Desktop on a Mac only mounts folders it has been told to share.
FROM ghcr.io/lichtblick-suite/lichtblick:latest
COPY default-layout.json /lichtblick/default-layout.json
