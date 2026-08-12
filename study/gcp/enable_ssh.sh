#!/bin/bash
set -euxo pipefail
systemctl enable --now ssh.service
