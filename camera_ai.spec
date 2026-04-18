# camera_ai.spec — PyInstaller build config cho Camera AI v2
# Chạy: pyinstaller camera_ai.spec
# ============================================================

import os
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

# Thu thập tự động data files từ customtkinter (themes, fonts)
ctk_datas = collect_data_files('customtkinter')

a = Analysis(
    ['gui.py'],                         # Entry point duy nhất (GUI)
    pathex=['.'],
    binaries=[],
    datas=[
        ('weights/best.pt', 'weights'), # Model AI — BẮT BUỘC
        ('.env', '.'),                  # Cấu hình môi trường
        *ctk_datas,                     # Assets của CustomTkinter (themes, fonts)
    ],
    hiddenimports=[
        'customtkinter',
        'PIL._tkinter_finder',
        'ultralytics',
        'ultralytics.nn.tasks',
        'ultralytics.nn.modules',
        'cv2',
        'requests',
        'dotenv',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'matplotlib',
        'notebook',
        'ipykernel',
        'IPython',
        'scipy',
        'pandas',
        'mkl',
        'test',
    ],
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
    name='CameraAI',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,          # ← Ẩn terminal đen — chỉ hiện cửa sổ GUI
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,              # Thêm 'icon.ico' nếu có icon riêng
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='CameraAI_v2',     # Tên thư mục output trong dist/
)
