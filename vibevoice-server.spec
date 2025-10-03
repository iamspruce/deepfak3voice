# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for VibeVoice Server
Based on your actual requirements.txt

Usage:
    pyinstaller vibevoice-server.spec --clean --noconfirm

Directory structure:
    /root
    ├── vibevoice/          (your module)
    ├── speak.py           (entry point)
    ├── requirements.txt
    └── vibevoice-server.spec (this file)
"""
import os
import sys
from PyInstaller.utils.hooks import collect_all, collect_submodules, collect_data_files

block_cipher = None

# Initialize collections
datas = []
hiddenimports = []
binaries = []

print("=" * 60)
print("Building VibeVoice Server with PyInstaller")
print("=" * 60)

# Core packages from requirements.txt that need full collection
core_packages = [
    # FastAPI and server
    'fastapi',
    'uvicorn',
    'starlette',
    'pydantic',
    'pydantic_core',
    
    # PyTorch and ML
    'torch',
    'torchaudio',
    'transformers',
    'diffusers',
    'huggingface_hub',
    'tokenizers',
    
    # Audio processing
    'librosa',
    'soundfile',
    'noisereduce',
    'scipy',
    
    # Data processing
    'numpy',
    'pandas',
    
    # API and networking
    'aiofiles',
    'sse_starlette',
]

print("\n📦 Collecting packages...")
for package in core_packages:
    try:
        pkg_datas, pkg_binaries, pkg_hiddenimports = collect_all(package)
        datas += pkg_datas
        binaries += pkg_binaries
        hiddenimports += pkg_hiddenimports
        print(f"  ✓ {package:<25} ({len(pkg_hiddenimports)} imports)")
    except Exception as e:
        print(f"  ⚠ {package:<25} Warning: {e}")

# Collect vibevoice module
print("\n📂 Collecting vibevoice module...")
if os.path.exists('vibevoice'):
    vibevoice_submodules = collect_submodules('vibevoice')
    hiddenimports += vibevoice_submodules
    print(f"  ✓ Found {len(vibevoice_submodules)} submodules")
else:
    print("  ⚠ WARNING: vibevoice directory not found!")

# Critical hidden imports for FastAPI/Uvicorn
print("\n🔧 Adding critical runtime imports...")
critical_imports = [
    # Uvicorn core
    'uvicorn.logging',
    'uvicorn.loops',
    'uvicorn.loops.auto',
    'uvicorn.protocols',
    'uvicorn.protocols.http',
    'uvicorn.protocols.http.auto',
    'uvicorn.protocols.http.h11_impl',
    'uvicorn.protocols.http.httptools_impl',
    'uvicorn.protocols.websockets',
    'uvicorn.protocols.websockets.auto',
    'uvicorn.protocols.websockets.wsproto_impl',
    'uvicorn.lifespan',
    'uvicorn.lifespan.on',
    
    # ASGI/async
    'anyio',
    'anyio._backends',
    'anyio._backends._asyncio',
    'sniffio',
    'h11',
    'httptools',
    'websockets',
    
    # Multipart
    'multipart',
    'multipart.multipart',
    
    # Email/MIME
    'email',
    'email.mime',
    'email.mime.multipart',
    'email.mime.text',
    'email.mime.base',
    
    # PyTorch internals
    'torch._C',
    'torch._dynamo',
    'torch.nn',
    'torch.nn.functional',
    'torch.nn.modules',
    'torchaudio.transforms',
    'torchaudio.functional',
    'torchaudio.backend',
    
    # Librosa/audio internals
    'librosa.core',
    'librosa.feature',
    'librosa.filters',
    'librosa.util',
    'soundfile',
    'soundfile._soundfile',
    
    # Scipy internals
    'scipy.signal',
    'scipy.fft',
    'scipy.special',
    'scipy.ndimage',
    'scipy.interpolate',
    
    # Numpy internals
    'numpy.core',
    'numpy.fft',
    'numpy.random',
    
    # Transformers/Diffusers internals
    'transformers.models',
    'transformers.modeling_utils',
    'transformers.tokenization_utils',
    'diffusers.models',
    'diffusers.schedulers',
    'diffusers.pipelines',
    
    # Additional utilities
    'PIL',
    'PIL._imaging',
    'requests',
    'urllib3',
    'certifi',
    'charset_normalizer',
]

hiddenimports += critical_imports
print(f"  ✓ Added {len(critical_imports)} critical imports")

# Add application files as data
print("\n📄 Adding application files...")
app_files = [
    ('vibevoice', 'vibevoice'),
    ('speak.py', '.'),
    ('requirements.txt', '.'),
]

for src, dst in app_files:
    if os.path.exists(src):
        datas.append((src, dst))
        print(f"  ✓ {src}")
    else:
        print(f"  ⚠ {src} not found!")

# Packages to exclude (reduce size)
excludes = [
    'matplotlib',
    'tkinter',
    'pytest',
    'IPython',
    'jupyter',
    'notebook',
    'sphinx',
    'setuptools._distutils',
    'distutils',
    'test',
    'tests',
]

print("\n🚫 Excluding unnecessary packages...")
print(f"  ✓ Excluding {len(excludes)} packages")

print("\n⚙️  Creating Analysis...")
a = Analysis(
    ['speak.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

print("📚 Creating PYZ archive...")
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

print("🔨 Creating executable...")
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='vibevoice-server',
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
    codesign_identity=None,
    entitlements_file=None,
    icon=None,  # Add path to .ico/.icns if you have one
)

print("\n" + "=" * 60)
print("✅ Build configuration complete!")
print("=" * 60)
print("\nNext steps:")
print("  1. Run: pyinstaller vibevoice-server.spec --clean")
print("  2. Find executable in: ./dist/vibevoice-server")
print("  3. Test: ./dist/vibevoice-server")
print("\n⚠️  Note: First run will download models (~1-2GB)")
print("=" * 60)