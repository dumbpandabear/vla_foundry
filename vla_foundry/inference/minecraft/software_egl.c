/* SDL2's offscreen driver enumerates at most eight EGL devices. On an
 * eight-GPU worker this truncates Mesa's ninth (software) device. Expose
 * only software devices to this SDL loader; all other EGL calls use GLVND.
 * No GPU device access or permissions are changed.
 */
#include <dlfcn.h>
#include <stdlib.h>
#include <string.h>

typedef unsigned int EGLBoolean;
typedef void *EGLDevice;
typedef void (*EGLProc)(void);
typedef EGLProc (*GetProc)(const char *);
typedef EGLBoolean (*QueryDevices)(int, EGLDevice *, int *);
typedef const char *(*QueryString)(EGLDevice, int);

static GetProc real_get_proc(void) {
    static GetProc get_proc;
    if (!get_proc) {
        void *egl = dlopen("libEGL.so.1", RTLD_NOW | RTLD_LOCAL);
        if (egl) get_proc = (GetProc)dlsym(egl, "eglGetProcAddress");
    }
    return get_proc;
}

static EGLBoolean software_devices(int capacity, EGLDevice *out, int *count) {
    GetProc get_proc = real_get_proc();
    if (!get_proc || !count || capacity < 0) return 0;
    QueryDevices query = (QueryDevices)get_proc("eglQueryDevicesEXT");
    QueryString query_string = (QueryString)get_proc("eglQueryDeviceStringEXT");
    int total = 0;
    if (!query || !query_string || !query(0, NULL, &total)) return 0;
    EGLDevice *devices = calloc(total ? total : 1, sizeof(*devices));
    if (!devices) return 0;
    if (!query(total, devices, &total)) {
        free(devices);
        return 0;
    }
    int found = 0;
    for (int i = 0; i < total; i++) {
        const char *extensions = query_string(devices[i], 0x3055);
        if (extensions && strstr(extensions, "EGL_MESA_device_software")) {
            if (out && found < capacity) out[found] = devices[i];
            found++;
        }
    }
    free(devices);
    *count = out ? (found < capacity ? found : capacity) : found;
    return 1;
}

EGLProc eglGetProcAddress(const char *name) {
    GetProc get_proc = real_get_proc();
    if (strcmp(name, "eglQueryDevicesEXT") == 0) return (EGLProc)software_devices;
    return get_proc ? get_proc(name) : NULL;
}
