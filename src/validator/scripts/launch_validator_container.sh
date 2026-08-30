#!/bin/bash
# Container entrypoint for the validator (k8s, prod only).
#
# Secrets arrive as plain env vars from the k8s Secret `validator-wallets`,
# synced from Vault by the Vault Secrets Operator (envs/prod/vso-validator.yaml).
# Two of them are whole bittensor key files as JSON; we materialise them on disk
# so main.py can keep using bt.Wallet(name, hotkey) exactly as before:
#
#   IOTA_VALIDATOR_COLDKEYPUB_JSON -> ~/.bittensor/wallets/iota/coldkeypub.txt
#   IOTA_VALIDATOR_HOTKEY_JSON     -> ~/.bittensor/wallets/iota/hotkeys/validator-0
#
# The wallet dir is an emptyDir, so this runs on every pod start.

set -e

source .venv/bin/activate

# Wallet identity is fixed for the prod validator; no secret needed for it.
export wallet_name="${wallet_name:-iota}"
export wallet_hotkey="${wallet_hotkey:-validator-0}"

WALLET_DIR="${HOME}/.bittensor/wallets/${wallet_name}"

if [ -z "${IOTA_VALIDATOR_HOTKEY_JSON:-}" ]; then
  echo "[-] IOTA_VALIDATOR_HOTKEY_JSON not set — Secret validator-wallets missing or empty" >&2
  exit 1
fi
mkdir -p "${WALLET_DIR}/hotkeys"
(umask 077 && printf '%s' "${IOTA_VALIDATOR_HOTKEY_JSON}" > "${WALLET_DIR}/hotkeys/${wallet_hotkey}")
echo "[+] wrote hotkey file ${WALLET_DIR}/hotkeys/${wallet_hotkey}"

if [ -n "${IOTA_VALIDATOR_COLDKEYPUB_JSON:-}" ]; then
  mkdir -p "${WALLET_DIR}"
  (umask 022 && printf '%s' "${IOTA_VALIDATOR_COLDKEYPUB_JSON}" > "${WALLET_DIR}/coldkeypub.txt")
  echo "[+] wrote ${WALLET_DIR}/coldkeypub.txt"
fi

# Never leak the key material into the validator process environment.
unset IOTA_VALIDATOR_HOTKEY_JSON IOTA_VALIDATOR_COLDKEYPUB_JSON

echo "[+] Launching validator (wallet=${wallet_name} hotkey=${wallet_hotkey})"
exec python main.py
