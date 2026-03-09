# -*- mode: python ; coding: utf-8 -*-
import os
from pathlib import Path
from dotenv import load_dotenv
from PyInstaller.utils.hooks import collect_dynamic_libs

load_dotenv()

MONOREPO_ROOT = Path(os.getcwd()).parents[1]
ATTEST_SO_PATH = MONOREPO_ROOT / "shared/common/src/common/attest.cpython-310-darwin.so"
ENTITLEMENTS_PATH = MONOREPO_ROOT / "scripts" / "entitlements.plist"
ENTITLEMENTS_PATH = Path(os.getcwd()).parents[1] / "scripts" / "entitlements.plist"

SIGNING_IDENTITY = os.getenv("MACOS_SIGNING_IDENTITY")
IROH_BINARIES = collect_dynamic_libs("iroh", destdir="iroh")

a = Analysis(
    ["main_pool.py"],
    pathex=[],
    binaries=[(str(ATTEST_SO_PATH), "common"), *IROH_BINARIES],
    datas=[],
    hiddenimports=["platformdirs.macos", "iroh.iroh_ffi"],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="main_pool",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=SIGNING_IDENTITY,
    entitlements_file=str(ENTITLEMENTS_PATH) if ENTITLEMENTS_PATH.exists() else None,
)
