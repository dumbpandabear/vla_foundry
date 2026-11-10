"""Select Mesa's software EGL device for SDL offscreen rendering on GPU workers."""

import ctypes
import hashlib
import os
import platform
import subprocess
from pathlib import Path


def _software_loader():
    source = Path(__file__).with_name("software_egl.c")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    cache = Path.home() / ".cache" / "minecraft-vla" / "renderer"
    cache.mkdir(parents=True, exist_ok=True)
    library = cache / f"software-egl-{platform.machine()}-{digest}.so"
    if not library.exists():
        temporary = cache / f"software-egl-{digest}-{os.getpid()}.so"
        # Link GLVND as a dependency so SDL's dlsym also finds standard EGL
        # entrypoints. Our eglGetProcAddress only filters device enumeration.
        subprocess.run(
            [
                "cc",
                "-shared",
                "-fPIC",
                "-O2",
                "-Wall",
                "-Werror",
                str(source),
                "-Wl,--no-as-needed",
                "-l:libEGL.so.1",
                "-ldl",
                "-o",
                str(temporary),
            ],
            check=True,
        )
        temporary.replace(library)
    return library


def select_software_egl_device():
    if os.environ.get("LIBGL_ALWAYS_SOFTWARE") != "1":
        return None
    egl = ctypes.CDLL("libEGL.so.1")
    egl.eglGetProcAddress.argtypes = [ctypes.c_char_p]
    egl.eglGetProcAddress.restype = ctypes.c_void_p
    query_devices = ctypes.CFUNCTYPE(
        ctypes.c_uint, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_int)
    )(egl.eglGetProcAddress(b"eglQueryDevicesEXT"))
    query_string = ctypes.CFUNCTYPE(ctypes.c_char_p, ctypes.c_void_p, ctypes.c_int)(
        egl.eglGetProcAddress(b"eglQueryDeviceStringEXT")
    )
    devices = (ctypes.c_void_p * 64)()
    count = ctypes.c_int()
    if not query_devices(64, devices, ctypes.byref(count)):
        raise RuntimeError("EGL device enumeration failed")
    for index in range(count.value):
        extensions = query_string(devices[index], 0x3055) or b""
        if b"EGL_MESA_device_software" in extensions.split():
            if index >= 8:
                os.environ["SDL_VIDEO_EGL_DRIVER"] = str(_software_loader())
                os.environ["SDL_HINT_EGL_DEVICE"] = "0"
            else:
                os.environ["SDL_HINT_EGL_DEVICE"] = str(index)
            print(f"Selected software EGL device {index} of {count.value}", flush=True)
            return index
    raise RuntimeError(f"No Mesa software EGL device found among {count.value} EGL devices")
