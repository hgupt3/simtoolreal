#!/bin/bash
set -euo pipefail

systemctl enable --now ssh.service

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
RuntimeMaxSec=7d
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

