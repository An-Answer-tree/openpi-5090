"""Task-specific camera profile used by the fixed six-view dataset."""

from __future__ import annotations

from collections.abc import Mapping
import dataclasses
import hashlib
import math
import pathlib
from typing import Any
import xml.etree.ElementTree as ET

from libero.libero.envs import TASK_MAPPING
import yaml

profile_sha256 = "6f7589700de0d1a071b62593137912878b1080d66ed8340ac009a77cd84d2346"
profile_path = pathlib.Path(__file__).with_name("tuned_camera_profiles") / "task_operation_cameras.yaml"
operation_camera_names = (
    "operation_backview",
    "operation_leftview",
    "operation_rightview",
    "operation_topview",
)
camera_injection_order = (
    "operation_topview",
    "operation_leftview",
    "operation_rightview",
    "operation_backview",
)
fovy_reference_order = ("frontview", "birdview", "sideview", "agentview")


def _float_tuple(value: Any, length: int, label: str) -> tuple[float, ...]:
    if not isinstance(value, list | tuple) or len(value) != length:
        raise ValueError(f"{label} must contain exactly {length} numbers")
    parsed = tuple(float(component) for component in value)
    if not all(math.isfinite(component) for component in parsed):
        raise ValueError(f"{label} must contain only finite numbers")
    return parsed


@dataclasses.dataclass(frozen=True)
class CameraPose:
    position: tuple[float, float, float]
    quaternion_wxyz: tuple[float, float, float, float]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> CameraPose:
        position = _float_tuple(value["position"], 3, "position")
        quaternion = _float_tuple(value["quaternion_wxyz"], 4, "quaternion_wxyz")
        norm = math.sqrt(sum(component * component for component in quaternion))
        return cls(position, tuple(component / norm for component in quaternion))

    def xml_position(self) -> list[float]:
        return [float(f"{component:.6f}") for component in self.position]

    def xml_quaternion(self) -> list[float]:
        return [float(f"{component:.6f}") for component in self.quaternion_wxyz]


class TunedCameraProfile:
    """Camera poses copied verbatim from the fixed-dataset generator."""

    def __init__(self, poses: Mapping[str, Any]) -> None:
        self._poses = poses

    @classmethod
    def load_bundled(cls) -> TunedCameraProfile:
        payload = profile_path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if digest != profile_sha256:
            raise ValueError(f"Tuned camera profile checksum mismatch: {digest}")
        return cls(yaml.safe_load(payload)["suites"])

    def poses_for_bddl(self, bddl_file_name: str | pathlib.Path) -> dict[str, CameraPose]:
        bddl_path = pathlib.Path(bddl_file_name)
        camera_poses = self._poses[bddl_path.parent.name][bddl_path.stem]
        return {name: CameraPose.from_mapping(camera_poses[name]) for name in operation_camera_names}


class TunedCameraInjector:
    """Inject fixed operation cameras without changing native cameras."""

    def __init__(self, profile: TunedCameraProfile) -> None:
        self._profile = profile

    def add_to_arena(self, bddl_file_name: str | pathlib.Path, mujoco_arena: Any) -> None:
        camera_map = self._camera_map(mujoco_arena)
        fovy = self._reference_fovy(camera_map)
        poses = self._profile.poses_for_bddl(bddl_file_name)

        for camera_name in camera_injection_order:
            if camera_name in camera_map:
                continue
            pose = poses[camera_name]
            camera_attributes = {"mode": "fixed"}
            if fovy is not None:
                camera_attributes["fovy"] = fovy
            mujoco_arena.set_camera(
                camera_name=camera_name,
                pos=pose.xml_position(),
                quat=pose.xml_quaternion(),
                camera_attribs=camera_attributes,
            )

    def _camera_map(self, mujoco_arena: Any) -> dict[str, ET.Element]:
        root = mujoco_arena.root if hasattr(mujoco_arena, "root") else mujoco_arena.worldbody
        return {camera.get("name"): camera for camera in root.iter("camera") if camera.get("name") is not None}

    def _reference_fovy(self, camera_map: Mapping[str, ET.Element]) -> str | None:
        for camera_name in fovy_reference_order:
            if camera_name in camera_map:
                return camera_map[camera_name].get("fovy")
        return None


class TunedCameraInstaller:
    """Install the fixed camera profile into LIBERO task classes once."""

    _installed = False

    @classmethod
    def install(cls) -> None:
        if cls._installed:
            return

        injector = TunedCameraInjector(TunedCameraProfile.load_bundled())
        for task_class in set(TASK_MAPPING.values()):
            original_setup_camera = task_class._setup_camera  # noqa: SLF001
            if getattr(original_setup_camera, "_openpi_tuned_camera_patched", False):
                continue

            def patched_setup_camera(self, mujoco_arena, _original_setup_camera=original_setup_camera):
                _original_setup_camera(self, mujoco_arena)
                injector.add_to_arena(self.bddl_file_name, mujoco_arena)

            patched_setup_camera._openpi_tuned_camera_patched = True  # noqa: SLF001
            task_class._setup_camera = patched_setup_camera  # noqa: SLF001

        cls._installed = True
