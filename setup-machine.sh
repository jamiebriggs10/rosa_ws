#!/usr/bin/env bash
# The once-only steps that need sudo on a fresh Ubuntu 22.04 computer, so that it can be
# the robot's computer: drivers, ROS 2 Humble, Docker, Ollama. Run it once as yourself
# (it asks for your password), then install.sh (which never needs sudo):
#
#   bash ~/rosa_ws/setup-machine.sh [--no-kernel] [--no-nvidia] [--no-docker] [--no-ollama] [--ssh] [--model NAME]
#
# Every step is skipped when it is already done, so running it twice changes nothing.
# It refuses to run on anything but Ubuntu 22.04, logs to ~/setup-machine.log, and ends
# by saying what it did and whether to reboot (needed after a new kernel or graphics driver).
# Read it top to bottom before you run it: one function per step, each with a note saying why.
set -euo pipefail

KERNEL=1 NVIDIA=1 DOCKER=1 OLLAMA=1 SSH=0
MODEL="llama3.1:8b"
while [ $# -gt 0 ]; do
  case "$1" in
    --no-kernel) KERNEL=0; shift ;;
    --no-nvidia) NVIDIA=0; shift ;;
    --no-docker) DOCKER=0; shift ;;
    --no-ollama) OLLAMA=0; shift ;;
    --ssh) SSH=1; shift ;;
    --model) MODEL="$2"; shift 2 ;;
    -h|--help) sed -n '2,11p' "$0"; exit 0 ;;
    *) echo "Unknown option: $1 (try --help)"; exit 2 ;;
  esac
done

LOG="$HOME/setup-machine.log"
ME="${SUDO_USER:-$USER}"          # the person's own login, even if someone ran this with sudo
SUDO="sudo"; [ "$(id -u)" = 0 ] && SUDO=""
NEED_REBOOT=0                     # set by the kernel and driver steps
DONE=()                           # one line per thing this run changed, for the summary
APT_UPDATED=0

say() { printf '\n==> %s\n' "$*"; }
fail() { printf '\nSetup stopped: %s\n' "$*" >&2; exit 1; }
installed() { dpkg -s "$1" >/dev/null 2>&1; }      # is this apt package installed?

apt_update() {
  # Refresh apt's package list once per run, and again after a new package source is added.
  $SUDO apt-get update -q
  APT_UPDATED=1
}

apt_install() {
  # Install only the packages in the list that are missing, so a second run installs nothing.
  local missing=() p
  for p in "$@"; do installed "$p" || missing+=("$p"); done
  if [ ${#missing[@]} -eq 0 ]; then echo "  already installed: $*"; return; fi
  [ "$APT_UPDATED" = 1 ] || apt_update
  DEBIAN_FRONTEND=noninteractive $SUDO apt-get install -y -q "${missing[@]}"
  DONE+=("installed ${missing[*]}")
}

check_ubuntu() {
  # Everything below uses package names and sources that are specific to Ubuntu 22.04
  # ("jammy"). OS_RELEASE can point at another file to test this check elsewhere.
  local release="${OS_RELEASE:-/etc/os-release}" id="" version=""
  if [ -f "$release" ]; then
    id="$(. "$release"; echo "${ID:-}")"
    version="$(. "$release"; echo "${VERSION_ID:-}")"
  fi
  if [ "$id" != "ubuntu" ] || [ "$version" != "22.04" ]; then
    fail "This script is for Ubuntu 22.04 (this computer: ${id:-unknown} ${version:-unknown})."
  fi
}

base_packages() {
  # The tools every later step and install.sh rely on: git to clone, curl to fetch installers,
  # tmux so the robot software outlives a closed terminal, Python's venv and pip for the agent,
  # build-essential for anything that compiles, net-tools for the usual network commands.
  apt_install git curl tmux python3-pip python3-venv build-essential net-tools libportaudio2 libsndfile1   # the last two: the microphone and WAV files (voice/)
}

hwe_kernel() {
  # The HWE ("hardware enablement") kernel is the newer Linux kernel Ubuntu 22.04 offers for
  # recent hardware. The original 5.15 kernel is from 2021 and does not know a 2025 workstation.
  apt_install linux-generic-hwe-22.04
  # A new kernel only runs after a reboot: compare the one running with the newest on disk.
  local newest; newest="$(ls /boot/vmlinuz-* | sed 's|/boot/vmlinuz-||' | sort -V | tail -1)"
  if [ "$(uname -r)" != "$newest" ]; then
    echo "  kernel $newest is installed but $(uname -r) is running: reboot needed"
    NEED_REBOOT=1
  else
    echo "  running the newest kernel ($newest)"
  fi
}

newest_nvidia_driver() {
  # The highest NNN among the nvidia-driver-NNN-open packages apt knows about, or nothing.
  apt-cache search --names-only '^nvidia-driver-[0-9]+-open$' | awk '{print $1}' | grep -o '[0-9]\+' | sort -n | tail -1
}

nvidia_driver() {
  # Only when there is an NVIDIA card. The "-open" driver is NVIDIA's open-source kernel
  # module, the one NVIDIA now recommends for recent cards; 570 is the first version that
  # supports them properly. Without it the screen runs on a slow fallback and Ollama has no GPU.
  if ! lspci 2>/dev/null | grep -qi nvidia; then echo "  no NVIDIA card found, skipping"; return; fi
  if nvidia-smi >/dev/null 2>&1; then
    echo "  driver already working (version $(nvidia-smi --query-gpu=driver_version --format=csv,noheader))"; return
  fi
  local version; version="$(newest_nvidia_driver)"
  if [ -z "$version" ] || [ "$version" -lt 570 ]; then
    # Ubuntu 22.04's own list stops before 570. The graphics-drivers PPA (a package source
    # kept by Ubuntu volunteers) carries the newer ones.
    command -v add-apt-repository >/dev/null || apt_install software-properties-common
    $SUDO add-apt-repository -y ppa:graphics-drivers/ppa
    apt_update
    version="$(newest_nvidia_driver)"
  fi
  { [ -n "$version" ] && [ "$version" -ge 570 ]; } || fail "no nvidia-driver-NNN-open with NNN >= 570 is on offer, even from the PPA"
  apt_install "nvidia-driver-$version-open"
  echo "  the driver loads after a reboot"
  NEED_REBOOT=1
}

ros2_humble() {
  # ROS 2 Humble is the robot middleware: the UR driver, MoveIt and our agent all talk
  # through it. Ubuntu does not ship it, so first add ROS's own package source (the
  # ros2-apt-source package from the ROS infrastructure team), then install the pieces:
  # the desktop set with RViz, the build tools, MoveIt, the Universal Robots driver,
  # the controller command line, and colcon (the workspace build tool).
  if ! installed ros2-apt-source; then
    command -v add-apt-repository >/dev/null || apt_install software-properties-common
    $SUDO add-apt-repository -y universe       # the Ubuntu section that ROS's dependencies live in
    local tag deb
    tag="$(curl -fsSL https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest | grep -F '"tag_name"' | awk -F'"' '{print $4}')"
    [ -n "$tag" ] || fail "could not read the latest ros2-apt-source release from GitHub (is the network up?)"
    deb="/tmp/ros2-apt-source.deb"
    curl -fsSL -o "$deb" "https://github.com/ros-infrastructure/ros-apt-source/releases/download/$tag/ros2-apt-source_${tag}.jammy_all.deb"
    $SUDO dpkg -i "$deb"
    apt_update
    DONE+=("added the ROS 2 package source ($tag)")
  fi
  apt_install ros-humble-desktop ros-dev-tools ros-humble-moveit ros-humble-ur ros-humble-ros2controlcli python3-colcon-common-extensions \
    ros-humble-foxglove-bridge                  # feeds the console's Lichtblick tab (the engineer's view)
}

docker_engine() {
  # Docker runs URSim, Universal Robots' own simulator, in a container. Installed from
  # Docker's own package source (docker-ce), as the lab asks, not Ubuntu's docker.io or
  # the snap. Being in the "docker" group lets you use it without sudo.
  if snap list docker >/dev/null 2>&1; then fail "Docker is installed as a snap; remove it first (sudo snap remove docker) so the Docker package can be used"; fi
  if installed docker.io; then fail "Ubuntu's docker.io package is installed; remove it first (sudo apt-get remove docker.io) so Docker's own package can be used"; fi
  if ! installed docker-ce; then
    $SUDO install -m 0755 -d /etc/apt/keyrings
    $SUDO curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
    $SUDO chmod a+r /etc/apt/keyrings/docker.asc
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu jammy stable" \
      | $SUDO tee /etc/apt/sources.list.d/docker.list >/dev/null
    apt_update
    DONE+=("added Docker's package source")
  fi
  apt_install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
  if id -nG "$ME" | grep -qw docker; then
    echo "  $ME is already in the docker group"
  else
    $SUDO usermod -aG docker "$ME"
    DONE+=("added $ME to the docker group (counts from the next login)")
  fi
  # Lichtblick's web build, the console's engineer's view; start.sh runs it. Pulled now so the
  # first start.sh does not wait for a download. Needs the docker group, so sudo if not yet in it.
  if $SUDO docker image inspect ghcr.io/lichtblick-suite/lichtblick:latest >/dev/null 2>&1; then
    echo "  the Lichtblick image is already here"
  else
    $SUDO docker pull -q ghcr.io/lichtblick-suite/lichtblick:latest && DONE+=("pulled the Lichtblick image")
  fi
}

ollama_server() {
  # Ollama serves the language model on this computer at http://localhost:11434. Its own
  # installer sets it up as a system service that starts with the computer and uses the
  # NVIDIA driver when one is loaded. The model is pulled now so the first command on the
  # console does not wait for a download of several gigabytes.
  if command -v ollama >/dev/null; then
    echo "  already installed ($(ollama --version 2>/dev/null | head -1))"
  else
    curl -fsSL https://ollama.com/install.sh | sh
    DONE+=("installed Ollama")
  fi
  local i; for i in $(seq 1 15); do curl -fs http://localhost:11434 >/dev/null 2>&1 && break; sleep 1; done
  if ollama list 2>/dev/null | grep -q "^$MODEL[[:space:]]"; then
    echo "  model $MODEL is already here"
  else
    ollama pull "$MODEL"
    DONE+=("pulled the model $MODEL")
  fi
}

ssh_server() {
  # Only with --ssh: lets another computer log in to this one over the network. Off by
  # default, because a lab computer should not accept logins unless that is wanted.
  if [ "$SSH" != 1 ]; then echo "  skipped (add --ssh to install openssh-server)"; return; fi
  apt_install openssh-server
  $SUDO systemctl enable -q --now ssh
}

summary() {
  say "Done. What this run changed:"
  if [ ${#DONE[@]} -eq 0 ]; then printf '  nothing: everything was already in place\n'; else printf '  %s\n' "${DONE[@]}"; fi
  echo
  if [ "$NEED_REBOOT" = 1 ]; then
    echo "Reboot now (sudo reboot). After that, run install.sh for the agent and the console."
  else
    echo "No reboot needed. Next: install.sh for the agent and the console."
  fi
  echo "Log of this run: $LOG"
}

check_ubuntu
exec > >(tee -a "$LOG") 2>&1       # everything below is shown and kept in the log
echo "setup-machine.sh on $(hostname) at $(date), as $ME"
[ -z "$SUDO" ] || sudo -v          # ask for the password once, up front

say "Base packages";    base_packages
say "HWE kernel";       if [ "$KERNEL" = 1 ]; then hwe_kernel; else echo "  skipped (--no-kernel)"; fi
say "NVIDIA driver";    if [ "$NVIDIA" = 1 ]; then nvidia_driver; else echo "  skipped (--no-nvidia)"; fi
say "ROS 2 Humble";     ros2_humble
say "Docker";           if [ "$DOCKER" = 1 ]; then docker_engine; else echo "  skipped (--no-docker)"; fi
say "Ollama";           if [ "$OLLAMA" = 1 ]; then ollama_server; else echo "  skipped (--no-ollama)"; fi
say "SSH server";       ssh_server
summary
