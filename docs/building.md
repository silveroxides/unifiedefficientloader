# Building the Native Extension

The `uel` native C library (`uel.dll` / `uel.so`) provides MMAP and CUDA memory management features. It must be compiled before `use_mmap=True` is available.

If the library is absent, `UnifiedSafetensorsLoader` falls back gracefully to standard IO for all operations.

## Requirements

### Windows

- Visual Studio 2019 or later with C++ workload
- CUDA Toolkit (12.x recommended) — `CUDA_PATH` environment variable must be set
- Internet access to download [Microsoft Detours](https://github.com/microsoft/Detours) (downloaded automatically)

### Linux

- GCC
- CUDA Toolkit — `CUDA_PATH` environment variable, or `nvcc` on `PATH`

## Build and install

Open an **x64 Native Tools Command Prompt for VS** (Windows) or a standard terminal (Linux).

### Build wheel and install

```bash
python setup.py build_ext bdist_wheel
pip install dist/unifiedefficientloader-*.whl --force-reinstall
```

### Force rebuild (ignore source timestamps)

```bash
python setup.py build_ext --force bdist_wheel
```

### Build extension only (for local development)

```bash
python setup.py build_ext
```

This compiles `uel.dll` / `uel.so` and copies it into `unifiedefficientloader/uel/` in the source tree. Combined with `pip install -e .` it makes the extension available immediately without reinstalling.

```bash
python setup.py build_ext
pip install -e .
```

## Incremental builds

The build system checks source file modification times against the existing output. If no `.c` or `.h` file is newer than `uel.dll` / `uel.so`, compilation is skipped:

```
uel.dll up-to-date, skipping. (pass --force to rebuild)
```

Pass `--force` to recompile regardless:

```bash
python setup.py build_ext --force
```

## Wheel tagging

The produced wheel is tagged `cp39-abi3-<platform>` (stable ABI). Since the extension is loaded via `ctypes` rather than as a Python C-extension module, it is not tied to any specific CPython version and runs on Python 3.9+.

```
dist/unifiedefficientloader-0.2.3-cp39-abi3-win_amd64.whl
```

## Build artifacts

All intermediate artifacts (`.obj`, `.lib`, `.exp`, Detours source) are placed in `build/temp.*` and are not committed to the repository. The `.gitignore` explicitly excludes:

```
unifiedefficientloader/uel/*.dll
unifiedefficientloader/uel/*.so
unifiedefficientloader/uel/*.lib
unifiedefficientloader/uel/*.exp
```

Only C and Python source files are tracked in version control.

## Verifying the build

```python
from unifiedefficientloader.uel import control

ok = control.init()
print("UEL loaded:", ok)
print("lib:", control.lib)
```

If `ok` is `True` and `lib` is not `None`, the native extension is loaded and ready.
