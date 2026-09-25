# -*- mode: python ; coding: utf-8 -*-

import os
import sys

block_cipher = None

added_files = [
    ('config', 'config'),
]

a = Analysis(
    ['run_3d_simulator.py'],
    pathex=['.'],
    binaries=[],
    datas=added_files,
    hiddenimports=[
        'dearpygui',
        'dearpygui.dearpygui',
        'cv2',
        'numpy',
        'scipy',
        'matplotlib',
        'yaml',
        'reportlab',
        'reportlab.lib',
        'reportlab.lib.pagesizes',
        'reportlab.platypus',
        'reportlab.lib.styles',
        'reportlab.lib.colors',
        'PIL',
        'jinja2',
        'simulator',
        'simulator.simulation',
        'simulator.renderer',
        'simulator.gui3d',
        'simulator.scenarios',
        'simulator.run_3d_sim',
        'ui',
        'ui.dashboard',
        'perception',
        'tracking',
        'control',
        'estimation',
        'camera',
        'sim',
        'core',
        'metrics',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='AstraTrack',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='AstraTrack',
)
