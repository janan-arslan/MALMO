# -*- mode: python ; coding: utf-8 -*-

block_cipher = None


a = Analysis(['preprocessing_combined_complete_ver_a.py'],
             pathex=['/Users/janan.arslan/Documents/MALMO/combined_3D_complete_pipeline'],
             binaries=[],
             datas=[],
             hiddenimports=[],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          [],
          name='MALMO_3D_Preprocess',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          upx_exclude=[],
          runtime_tmpdir=None,
          console=False , icon='malmo_vector_logo.icns')
app = BUNDLE(exe,
             name='MALMO_3D_Preprocess.app',
             icon='malmo_vector_logo.icns',
             bundle_identifier=None)
