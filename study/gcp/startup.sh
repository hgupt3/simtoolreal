#!/bin/bash
set -euo pipefail

systemctl enable --now ssh.service

# A source image may contain a newer Debian kernel than the kernel that was
# running when NVIDIA DKMS last built its modules. Repair that normal image-boot
# mismatch before Isaac Gym starts; otherwise CUDA reports zero devices and the
# simulator can segfault during initialization.
if ! nvidia-smi >/dev/null 2>&1; then
  systemctl stop simtoolreal-study.service 2>/dev/null || true
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y "linux-headers-$(uname -r)"
  /usr/sbin/dkms autoinstall -k "$(uname -r)"
  depmod -a
  modprobe nvidia
  systemctl restart nvidia-persistenced.service
fi

cat >/etc/systemd/system/simtoolreal-study.service <<'EOF'
[Unit]
Description=SimToolReal transformer study
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=tylerlum
WorkingDirectory=/home/tylerlum/simtoolreal
ExecStart=/usr/bin/bash /home/tylerlum/simtoolreal/study/gcp/run_worker.sh
Restart=on-failure
RestartSec=60
KillMode=control-group
KillSignal=SIGINT
TimeoutStopSec=120

[Install]
WantedBy=multi-user.target
EOF

cat >/etc/systemd/system/simtoolreal-study-sync.service <<'EOF'
[Unit]
Description=Sync SimToolReal study state to GCS

[Service]
Type=oneshot
User=tylerlum
WorkingDirectory=/home/tylerlum/simtoolreal
ExecStart=/usr/bin/bash /home/tylerlum/simtoolreal/study/gcp/sync_worker.sh
EOF

cat >/etc/systemd/system/simtoolreal-study-sync.timer <<'EOF'
[Unit]
Description=Sync and heartbeat SimToolReal study every ten minutes

[Timer]
OnBootSec=2min
OnUnitActiveSec=10min
Persistent=true

[Install]
WantedBy=timers.target
EOF

chmod +x /home/tylerlum/simtoolreal/study/gcp/run_worker.sh
chmod +x /home/tylerlum/simtoolreal/study/gcp/sync_worker.sh
systemctl daemon-reload
systemctl enable --now simtoolreal-study-sync.timer
systemctl enable --now simtoolreal-study.service
