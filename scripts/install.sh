#!/bin/bash
# Zak-OTFS Waveform Studio one-shot installer (oneclick/simdoc-style).
#
# Runs on Ubuntu/Debian or RHEL-family hosts. Creates the otfs user, lays
# down the repo into /opt/otfs, builds a Python venv with numpy + scipy +
# flask, installs + starts the systemd unit on port 8050. Idempotent —
# safe to re-run; the Update button in the UI re-runs this same script
# from the latest GitHub tarball.
#
# Usage:
#   sudo bash scripts/install.sh
#
# Override the listen port (default 8050) by exporting OTFS_PORT first.

set -euo pipefail

if [[ "${EUID}" -ne 0 ]]; then
    echo "ERROR: install.sh must run as root (use sudo)" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_ROOT="$(dirname "${SCRIPT_DIR}")"
[[ -f "${SRC_ROOT}/webui/app.py" ]] || {
    echo "ERROR: expected ${SRC_ROOT}/webui/app.py; running from wrong dir?" >&2
    exit 1
}

# ---- Knobs -----------------------------------------------------------------
SERVICE_USER="${OTFS_USER:-otfs}"
SERVICE_GROUP="${OTFS_GROUP:-otfs}"
OTFS_HOME="${OTFS_HOME:-/var/lib/otfs}"
INSTALL_DIR="${OTFS_INSTALL_DIR:-/opt/otfs}"
LISTEN_PORT="${OTFS_PORT:-8050}"
SYSTEMD_UNIT="/etc/systemd/system/otfs.service"
UPDATE_TARBALL_URL_DEFAULT="https://github.com/nikhilsimnovus/otfs/archive/refs/heads/main.tar.gz"

log()  { printf '\033[1;34m[install]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$*"; }
fail() { printf '\033[1;31m[fail]\033[0m %s\n' "$*" >&2; exit 1; }

# ---- 1) OS prereqs ----------------------------------------------------------
if   command -v apt-get >/dev/null 2>&1; then PKG=apt
elif command -v dnf     >/dev/null 2>&1; then PKG=dnf
elif command -v yum     >/dev/null 2>&1; then PKG=yum
else fail "no supported package manager (apt-get / dnf / yum)"
fi
log "Using ${PKG}"
case "${PKG}" in
  apt)
    DEBIAN_FRONTEND=noninteractive apt-get update -qq
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
        bash python3 python3-venv python3-pip curl tar ca-certificates
    ;;
  dnf|yum)
    ${PKG} install -y -q bash python3 python3-pip curl tar ca-certificates
    ;;
esac

# ---- 2) Service user + dirs -------------------------------------------------
if id -u "${SERVICE_USER}" >/dev/null 2>&1; then
    log "User ${SERVICE_USER} already exists"
else
    log "Creating user ${SERVICE_USER} (home=${OTFS_HOME})"
    useradd --system --create-home --home "${OTFS_HOME}" --shell /bin/bash "${SERVICE_USER}"
fi
install -d -m 0755 -o "${SERVICE_USER}" -g "${SERVICE_GROUP}" "${INSTALL_DIR}"

# ---- 3) Copy files ----------------------------------------------------------
log "Installing Zak-OTFS Waveform Studio to ${INSTALL_DIR}"
for sub in otfs webui scripts; do
    rm -rf "${INSTALL_DIR:?}/${sub}"
    cp -r "${SRC_ROOT}/${sub}" "${INSTALL_DIR}/${sub}"
done
cp "${SRC_ROOT}/requirements.txt" "${SRC_ROOT}/README.md" "${INSTALL_DIR}/" 2>/dev/null || true
find "${INSTALL_DIR}" -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
chown -R "${SERVICE_USER}:${SERVICE_GROUP}" "${INSTALL_DIR}"

# ---- 4) Python venv ----------------------------------------------------------
if [[ -x "${INSTALL_DIR}/venv/bin/python" ]]; then
    log "Venv exists at ${INSTALL_DIR}/venv — upgrading deps"
else
    log "Creating Python venv at ${INSTALL_DIR}/venv"
    sudo -u "${SERVICE_USER}" python3 -m venv "${INSTALL_DIR}/venv"
fi

pip_install() {
    local logf; logf="$(mktemp)"
    if sudo -u "${SERVICE_USER}" -E "${INSTALL_DIR}/venv/bin/pip" install --quiet "$@" >"$logf" 2>&1; then
        rm -f "$logf"; return 0
    fi
    if grep -qE 'CERTIFICATE_VERIFY_FAILED|SSLError|self-signed certificate' "$logf"; then
        warn "pip TLS verification failed (SSL-inspecting proxy?) — retrying with --trusted-host"
        if sudo -u "${SERVICE_USER}" -E "${INSTALL_DIR}/venv/bin/pip" install --quiet \
                --trusted-host pypi.org --trusted-host files.pythonhosted.org \
                --trusted-host pypi.python.org "$@" >"$logf" 2>&1; then
            rm -f "$logf"; return 0
        fi
    fi
    echo "----- pip output (last 30 lines) -----" >&2
    tail -30 "$logf" >&2; rm -f "$logf"
    fail "pip install failed for: $*"
}
log "Installing numpy + scipy + flask into venv"
pip_install --upgrade pip
pip_install --upgrade numpy scipy flask

# ---- 5) systemd unit ----------------------------------------------------------
log "Installing systemd unit -> ${SYSTEMD_UNIT}"
cat > "${SYSTEMD_UNIT}" <<UNIT
[Unit]
Description=Zak-OTFS Waveform Studio (Flask)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=${SERVICE_USER}
Group=${SERVICE_GROUP}
WorkingDirectory=${INSTALL_DIR}
Environment=HOME=${OTFS_HOME}
Environment=USER=${SERVICE_USER}
Environment=OTFS_PORT=${LISTEN_PORT}
ExecStart=${INSTALL_DIR}/venv/bin/python ${INSTALL_DIR}/webui/app.py
Restart=on-failure
RestartSec=3
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
UNIT

# ---- 5b) Self-update plumbing --------------------------------------------------
# The UI's Update button (POST /api/update) re-runs this installer from the
# latest GitHub tarball. The service user has no sudo by default, so plant:
#   * /usr/local/sbin/otfs-update — downloads the tarball, re-runs install.sh
#   * /etc/sudoers.d/otfs         — NOPASSWD entry for *only* that script
UPDATER_PATH="/usr/local/sbin/otfs-update"
log "Installing self-update helper -> ${UPDATER_PATH}"
cat > "${UPDATER_PATH}" <<UPDATER
#!/bin/bash
# Auto-generated by otfs install.sh. Triggered by the Update button in
# the Flask UI (POST /api/update). Downloads the latest tarball from the
# otfs GitHub repo and re-runs scripts/install.sh from it.
set -euo pipefail
TARBALL_URL="\${OTFS_UPDATE_TARBALL:-${UPDATE_TARBALL_URL_DEFAULT}}"
TD=\$(mktemp -d -p /tmp otfs-update-XXXXXX)
trap 'rm -rf "\$TD"' EXIT
echo "[otfs-update] downloading \$TARBALL_URL"
if ! curl -fsSL "\$TARBALL_URL" -o "\$TD/main.tar.gz" 2>"\$TD/curl.err"; then
    if grep -qiE 'self.signed|certificate|SSL' "\$TD/curl.err"; then
        echo "[otfs-update] SSL verification failed (corporate proxy?) — retrying with -k" >&2
        curl -fkSL "\$TARBALL_URL" -o "\$TD/main.tar.gz"
    else
        cat "\$TD/curl.err" >&2
        exit 1
    fi
fi
tar xzf "\$TD/main.tar.gz" -C "\$TD" --strip-components=1
exec bash "\$TD/scripts/install.sh"
UPDATER
chmod 0755 "${UPDATER_PATH}"

SUDOERS_FILE="/etc/sudoers.d/otfs"
log "Granting ${SERVICE_USER} passwordless sudo for ${UPDATER_PATH} only"
cat > "${SUDOERS_FILE}" <<SUDO
# Auto-generated by otfs install.sh. Lets the otfs service trigger a
# self-update via the Update button WITHOUT general sudo.
${SERVICE_USER} ALL=(root) NOPASSWD: ${UPDATER_PATH}
SUDO
chmod 0440 "${SUDOERS_FILE}"
if ! visudo -c -f "${SUDOERS_FILE}" >/dev/null 2>&1; then
    warn "sudoers entry has a syntax issue — removing to keep sudo working"
    rm -f "${SUDOERS_FILE}"
fi

systemctl daemon-reload
log "Enabling + starting otfs"
systemctl enable --now otfs.service
systemctl restart otfs.service
sleep 2
if systemctl is-active --quiet otfs.service; then
    log "otfs is ACTIVE"
else
    warn "otfs failed to start — see: journalctl -u otfs -n 50"
    systemctl status otfs --no-pager || true
    exit 1
fi

HOSTNAME_BEST=$(hostname -I 2>/dev/null | awk '{print $1}')
[[ -z "${HOSTNAME_BEST}" ]] && HOSTNAME_BEST="<host-ip>"
cat <<DONE

============================================================
  Zak-OTFS Waveform Studio is installed and running.
============================================================

  URL:      http://${HOSTNAME_BEST}:${LISTEN_PORT}/
  Service:  systemctl status otfs
  Logs:     journalctl -u otfs -f

Uninstall:
  sudo systemctl disable --now otfs
  sudo rm -f ${SYSTEMD_UNIT} /etc/sudoers.d/otfs /usr/local/sbin/otfs-update
  sudo rm -rf ${INSTALL_DIR} ${OTFS_HOME}
  sudo userdel ${SERVICE_USER}
  sudo systemctl daemon-reload
DONE
