# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['discriminant_analysis_gui.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['matplotlib', 'IPython', 'ipykernel', 'jupyter', 'PIL', 'pytz', 'dateutil', 'openpyxl', 'xlrd', 'xlwt', 'scipy.stats', 'scipy.spatial', 'scipy.optimize', 'scipy.integrate'],
    noarchive=False,
    optimize=1,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='discriminant_analysis_gui',
    debug=False,
    bootloader_ignore_signals=False,
    strip=True,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
