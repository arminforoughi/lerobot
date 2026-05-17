from .configuration_oakd import OAKDCameraConfig

__all__ = ["OAKDCamera", "OAKDCameraConfig"]


def __getattr__(name: str):
    # Lazy import: importing `depthai` can crash in some envs (e.g. duplicate glog),
    # but configs should remain importable so CLI `--help` and non-OAK workflows work.
    if name == "OAKDCamera":
        from .camera_oakd import OAKDCamera

        return OAKDCamera
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
