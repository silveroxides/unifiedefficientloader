import os
import sys
import re
import subprocess
import platform
import urllib.request
import zipfile
import shutil
from pathlib import Path
import setuptools
import wheel.bdist_wheel
from setuptools import setup, Distribution, Extension
from setuptools.command.build_ext import build_ext


def _read_pyproject_version():
    """Read version from pyproject.toml without importing any build tools."""
    pyproject = Path(__file__).parent / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")
    m = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    if not m:
        raise RuntimeError("Could not find version in pyproject.toml")
    return m.group(1)


class BuildUELExtension(build_ext):
    """
    Custom build_ext that compiles the raw C shared library (uel.dll / uel.so)
    into build_lib, keeping all intermediate artifacts (*.obj, *.lib, *.exp)
    inside build_temp so the source tree stays clean.

    Triggered by:
        python setup.py build_ext
        python setup.py bdist_wheel
        pip install .
    """

    def run(self):
        system = platform.system()
        if system not in ("Windows", "Linux"):
            raise RuntimeError(
                f"UEL native extension only supported on Windows and Linux (got {system})"
            )
        self._build()

    # ------------------------------------------------------------------
    # Platform helpers
    # ------------------------------------------------------------------

    def _cuda_path_win(self):
        p = os.environ.get("CUDA_PATH")
        if not p:
            raise RuntimeError(
                "CUDA_PATH environment variable not set. Install CUDA Toolkit."
            )
        return Path(p)

    def _cuda_path_linux(self):
        p = os.environ.get("CUDA_PATH", "/usr/local/cuda")
        if not Path(p).exists():
            try:
                nvcc = subprocess.check_output(["which", "nvcc"], text=True).strip()
                p = Path(nvcc).parent.parent
            except Exception:
                raise RuntimeError(
                    f"CUDA not found at {p} and nvcc not in PATH. Set CUDA_PATH."
                )
        return Path(p)

    def _vcvars(self):
        vswhere = (
            Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"))
            / "Microsoft Visual Studio"
            / "Installer"
            / "vswhere.exe"
        )
        if not vswhere.exists():
            raise RuntimeError("vswhere.exe not found. Install Visual Studio.")
        result = subprocess.run(
            [str(vswhere), "-latest", "-property", "installationPath"],
            capture_output=True,
            text=True,
            check=True,
        )
        vs = result.stdout.strip()
        if not vs:
            raise RuntimeError("Visual Studio not found.")
        vcvars = Path(vs) / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
        if not vcvars.exists():
            raise RuntimeError(f"vcvars64.bat not found at {vcvars}")
        return str(vcvars)

    def _ensure_detours(self, build_temp: Path) -> Path:
        detours_dir = build_temp / "Detours"
        lib_marker = detours_dir / "lib.X64" / "detours.lib"
        if lib_marker.exists():
            return detours_dir

        print("Downloading Microsoft Detours...")
        zip_path = build_temp / "detours.zip"
        urllib.request.urlretrieve(
            "https://github.com/microsoft/Detours/archive/refs/heads/master.zip",
            zip_path,
        )
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(build_temp)
        # GitHub zips extract to <repo>-<branch>, find whatever was extracted
        candidates = [
            build_temp / "Detours-master",
            build_temp / "Detours-main",
        ]
        extracted = next((p for p in candidates if p.exists()), None)
        if extracted is None:
            # fallback: find any Detours-* directory
            matches = list(build_temp.glob("Detours-*"))
            if not matches:
                raise RuntimeError("Could not find extracted Detours directory")
            extracted = matches[0]
        if detours_dir.exists():
            shutil.rmtree(detours_dir)
        shutil.move(str(extracted), str(detours_dir))

        print("Building Detours...")
        subprocess.check_call(
            "nmake",
            cwd=str(detours_dir / "src"),
            shell=True,
        )
        return detours_dir

    # ------------------------------------------------------------------
    # Staleness check
    # ------------------------------------------------------------------

    @staticmethod
    def _needs_rebuild(src_files: list, output: Path) -> bool:
        """Return True if any source file is newer than output, or output missing."""
        if not output.exists():
            return True
        out_mtime = output.stat().st_mtime
        return any(f.stat().st_mtime > out_mtime for f in src_files)

    # ------------------------------------------------------------------
    # Main build
    # ------------------------------------------------------------------

    def _build(self):
        build_temp = Path(self.build_temp).resolve()
        build_temp.mkdir(parents=True, exist_ok=True)

        # DLL/SO must land in both:
        #   1. source package dir (so editable installs work + staleness check)
        #   2. build_lib (so bdist_wheel/install_lib picks it up into the wheel)
        pkg_uel_dir = Path("unifiedefficientloader/uel").resolve()

        # build_lib is set by setuptools (e.g. build/lib.linux-x86_64-cpython-39)
        build_lib_uel_dir = Path(self.build_lib) / "unifiedefficientloader" / "uel"
        build_lib_uel_dir.mkdir(parents=True, exist_ok=True)

        src_root = Path("unifiedefficientloader/uel/src").resolve()
        system = platform.system()

        if system == "Windows":
            self._build_windows(build_temp, pkg_uel_dir, src_root)
            shutil.copy2(pkg_uel_dir / "uel.dll", build_lib_uel_dir / "uel.dll")
        else:
            self._build_linux(build_temp, pkg_uel_dir, src_root)
            shutil.copy2(pkg_uel_dir / "uel.so", build_lib_uel_dir / "uel.so")

    def _build_windows(self, build_temp: Path, pkg_uel_dir: Path, src_root: Path):
        src_win = Path("unifiedefficientloader/uel/src-win").resolve()
        dest = pkg_uel_dir / "uel.dll"

        src_files = list(src_root.glob("*.c")) + list(src_win.glob("*.c"))
        src_files += list(src_root.glob("*.h"))  # headers trigger rebuild too

        if not self.force and not self._needs_rebuild(src_files, dest):
            print("uel.dll up-to-date, skipping. (pass --force to rebuild)")
            return

        # Compile everything into build_temp — no artifacts touch the source tree
        tmp_dll = build_temp / "uel.dll"
        tmp_lib = build_temp / "uel.lib"  # /IMPLIB redirect

        detours_dir = self._ensure_detours(build_temp)
        cuda_path = self._cuda_path_win()

        src_str = " ".join(f'"{f}"' for f in src_files if f.suffix == ".c")

        includes = (
            f'/I"{src_root}" /I"{cuda_path / "include"}" /I"{detours_dir / "include"}"'
        )
        link_flags = (
            f'/LIBPATH:"{cuda_path / "lib" / "x64"}"'
            f' /LIBPATH:"{detours_dir / "lib.X64"}"'
            f' /IMPLIB:"{tmp_lib}"'
            f" cuda.lib cudart.lib dxgi.lib dxguid.lib detours.lib onecore.lib"
        )

        cmd = f'cl.exe /LD /O2 {src_str} {includes} /Fe"{tmp_dll}" /link {link_flags}'
        print("Building uel.dll...")
        subprocess.check_call(cmd, cwd=str(build_temp), shell=True)

        shutil.copy2(tmp_dll, dest)
        print(f"Installed: {dest}")

    def _build_linux(self, build_temp: Path, pkg_uel_dir: Path, src_root: Path):
        src_posix = Path("unifiedefficientloader/uel/src-posix").resolve()
        dest = pkg_uel_dir / "uel.so"

        src_files = list(src_root.glob("*.c")) + list(src_posix.glob("*.c"))
        src_files += list(src_root.glob("*.h"))

        if not self.force and not self._needs_rebuild(src_files, dest):
            print(f"uel.so up-to-date, skipping. (pass --force to rebuild)")
            return

        tmp_so = build_temp / "uel.so"
        cuda_path = self._cuda_path_linux()

        src_str = " ".join(f'"{f}"' for f in src_files if f.suffix == ".c")

        lib_dir = cuda_path / "lib64"
        lib_stubs = cuda_path / "targets" / "x86_64-linux" / "lib" / "stubs"
        lib_paths = f'-L"{lib_dir}"'
        if lib_stubs.exists():
            lib_paths += f' -L"{lib_stubs}"'

        cmd = (
            f"gcc -shared -fPIC -O2 -g {src_str}"
            f' -I"{src_root}" -I"{cuda_path / "include"}"'
            f" {lib_paths} -lcuda"
            f' -o "{tmp_so}"'
        )
        print("Building uel.so...")
        subprocess.check_call(cmd, cwd=str(build_temp), shell=True)

        shutil.copy2(tmp_so, dest)
        print(f"Installed: {dest}")


class BinaryDistribution(Distribution):
    """Tag the wheel as platform-specific (not pure Python)."""

    def has_ext_modules(self):
        return True


# Minimum Python version for abi3 stable ABI tag (must match requires-python)
_ABI3_MIN_PYTHON = (3, 9)


class BdistWheel(wheel.bdist_wheel.bdist_wheel):
    """
    bdist_wheel subclass that:
    - Accepts --force and forwards it to build_ext.
    - Tags the wheel as abi3 (cp39-abi3-<plat>) since the extension is a raw
      ctypes-loaded DLL/SO and is not tied to any specific CPython ABI version.

    Allows:
        python setup.py bdist_wheel --force
        python setup.py build_ext bdist_wheel --force
    """

    user_options = wheel.bdist_wheel.bdist_wheel.user_options + [
        ("force", "f", "forcibly rebuild C extension (ignore timestamps)"),
    ]
    boolean_options = wheel.bdist_wheel.bdist_wheel.boolean_options + ["force"]

    def initialize_options(self):
        super().initialize_options()
        self.force = False
        # Tell bdist_wheel to use the stable abi3 tag
        self.python_tag = "cp{}{}".format(*_ABI3_MIN_PYTHON)
        self.py_limited_api = "cp{}{}".format(*_ABI3_MIN_PYTHON)

    def run(self):
        if self.force:
            build_ext_cmd = self.get_finalized_command("build_ext")
            build_ext_cmd.force = True
        super().run()

    def get_tag(self):
        _impl, _abi, plat = super().get_tag()
        impl = "cp{}{}".format(*_ABI3_MIN_PYTHON)
        abi = "abi3"
        return impl, abi, plat


setup(
    name="unifiedefficientloader",
    version=_read_pyproject_version(),
    distclass=BinaryDistribution,
    cmdclass={
        "build_ext": BuildUELExtension,
        "bdist_wheel": BdistWheel,
    },
    # A real Extension object causes setuptools to invoke build_ext properly.
    # sources=[] because our custom run() handles all compilation manually.
    ext_modules=[
        Extension(
            name="unifiedefficientloader.uel.uel",
            sources=[],
        )
    ],
    packages=["unifiedefficientloader", "unifiedefficientloader.uel"],
    package_data={"unifiedefficientloader.uel": ["*.dll", "*.so"]},
    include_package_data=True,
)
