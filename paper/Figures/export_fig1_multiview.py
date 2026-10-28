"""Render synchronized Figure 1 views from the recorded Spatial episode state.

Use the LIBERO Conda environment, with MUJOCO_GL and the rendering device
configured before running. For the local CPU renderer, use Mesa EGL device 6:
LIBGL_ALWAYS_SOFTWARE=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
__EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json
MUJOCO_EGL_DEVICE_ID=6 python paper/Figures/export_fig1_multiview.py
"""

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from robosuite.utils.binding_utils import MjRenderContextOffscreen
from robosuite.utils.binding_utils import MjSim

SOURCE = Path(
    "/opt/liutong/libero_multiview_tuned_6view_hdf5/libero_spatial/"
    "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo.hdf5"
)
VIEWS = ("backview", "leftview", "rightview", "topview")


def main():
    """Restore one state without stepping physics and render four original cameras."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step", type=int, default=43)
    parser.add_argument("--resolution", type=int, default=1536)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "Figure1_views")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(SOURCE, "r") as dataset:
        episode = dataset["data/demo_0"]
        model_xml = episode.attrs["model_file"]
        state = episode["states"][args.step]
        sim = MjSim.from_xml_string(model_xml)
        sim.set_state_from_flattened(state)
        sim.forward()
        context = MjRenderContextOffscreen(sim, device_id=-1)
        context.vopt.geomgroup[0] = 0
        context.vopt.geomgroup[1] = 1

        manifest = {
            "source_hdf5": str(SOURCE),
            "suite": "libero_spatial",
            "task_index_zero_based": 0,
            "task": "pick up the black bowl between the plate and the ramekin and place it on the plate",
            "episode": "demo_0",
            "episode_frames": int(episode["actions"].shape[0]),
            "step_index_zero_based": args.step,
            "state_sha256": hashlib.sha256(state.tobytes()).hexdigest(),
            "model_xml_sha256": hashlib.sha256(model_xml.encode()).hexdigest(),
            "selection": "Approach to the target bowl; contrasting visibility across synchronized views.",
            "rendering": "Recorded XML and state; original cameras; no physics steps; vertical flip to image coordinates.",
            "resolution": [args.resolution, args.resolution],
            "images": [],
        }
        for view in VIEWS:
            camera = f"operation_{view}"
            native = sim.render(width=256, height=256, camera_name=camera)[::-1].copy()
            reference = episode[f"obs/{camera}_rgb"][args.step]
            error = float(np.abs(native.astype(float) - reference).mean())
            frame = sim.render(width=args.resolution, height=args.resolution, camera_name=camera)[::-1].copy()
            filename = f"{view}.png"
            Image.fromarray(frame).save(args.output_dir / filename, dpi=(300, 300))
            manifest["images"].append(
                {
                    "view": view,
                    "camera": camera,
                    "file": filename,
                    "native_256px_mean_absolute_error_0_255": error,
                    "sha256": hashlib.sha256((args.output_dir / filename).read_bytes()).hexdigest(),
                }
            )
            print(f"{view}: {args.resolution} x {args.resolution}; native reference MAE {error:.4f}", flush=True)
        sim.free()
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
