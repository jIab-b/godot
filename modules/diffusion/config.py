def can_build(env, platform):
    # The diffusion module requires RenderingDevice support
    # Currently available on desktop and mobile platforms
    return platform in ["windows", "linuxbsd", "macos", "android", "ios"]

def configure(env):
    pass

def get_doc_classes():
    return [
        "DiffusionEffect",
    ]

def get_doc_path():
    return "doc_classes"
