"""Export actual camera poses and a geometric guide for the layout illustration.

Run with the LIBERO Conda environment. This restores one recorded state;
it does not step physics or allocate a rendering context.
"""

import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from robosuite.utils.binding_utils import MjSim

SOURCE = Path(
    "/opt/liutong/libero_multiview_tuned_6view_hdf5/libero_spatial/"
    "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo.hdf5"
)
CAMERAS = {
    "Back": ("operation_backview", "#0072B2"),
    "Left": ("operation_leftview", "#E69F00"),
    "Right": ("operation_rightview", "#009E73"),
    "Top": ("operation_topview", "#CC79A7"),
    "Agent": ("agentview", "#374151"),
    "Wrist": ("robot0_eye_in_hand", "#374151"),
}


def main():
    """Save a coordinate-derived diagram and the underlying poses."""
    plt.switch_backend("Agg")
    output = Path(__file__).resolve().parent
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif"], "font.size": 12})
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_axes([0, 0.01, 1, 0.98], projection="3d", proj_type="ortho")
    table = np.array([[-0.8, -0.6, 0.875], [0.5, -0.6, 0.875], [0.5, 0.6, 0.875], [-0.8, 0.6, 0.875]])
    ax.add_collection3d(Poly3DCollection([table], facecolors="#F0F0F0", edgecolors="#999999", alpha=0.7))
    for corner in table:
        ax.plot([corner[0], corner[0]], [corner[1], corner[1]], [0.35, corner[2]], color="#BBBBBB", linewidth=2)

    geometry = {"source_hdf5": str(SOURCE), "episode": "demo_0", "step_index_zero_based": 43, "cameras": {}}
    with h5py.File(SOURCE, "r") as dataset:
        sim = MjSim.from_xml_string(dataset["data/demo_0"].attrs["model_file"])
        sim.set_state_from_flattened(dataset["data/demo_0/states"][43])
        sim.forward()
        joints = np.array([sim.data.body_xpos[sim.model.body_name2id(f"robot0_link{i}")] for i in range(8)])
        ax.plot(*joints.T, color="#777777", linewidth=9, solid_capstyle="round")
        ax.scatter(*joints.T, color="#333333", s=70, depthshade=False)
        ax.text(-0.74, -0.05, 0.76, "Robot base", fontsize=11)
        for label, (name, color) in CAMERAS.items():
            idx = sim.model.camera_name2id(name)
            position = sim.data.cam_xpos[idx].copy()
            forward = -sim.data.cam_xmat[idx].reshape(3, 3)[:, 2]
            plane_hit = position + (0.9 - position[2]) / forward[2] * forward
            geometry["cameras"][label] = {
                "name": name,
                "world_position": position.tolist(),
                "world_forward": forward.tolist(),
                "intersection_with_z_0_9_plane": plane_hit.tolist(),
                "fovy_degrees": float(sim.model.cam_fovy[idx]),
            }
            ray_length = np.linalg.norm(plane_hit - position)
            ax.plot(*np.array([position, plane_hit]).T, color=color, linewidth=1.1, linestyle="--", alpha=0.75)
            ax.quiver(*position, *forward, length=ray_length * 0.3, color=color, linewidth=2, arrow_length_ratio=0.18)
            ax.scatter(*position, color=color, s=90, marker="s", depthshade=False)
            offset = np.array([0, 0, 0.10]) if label != "Wrist" else np.array([0.07, 0.02, 0.04])
            ax.text(*(position + offset), label, color=color, fontsize=16, fontweight="bold", ha="center")

    ax.set_xlim(-1.1, 1.0)
    ax.set_ylim(-1.4, 1.5)
    ax.set_zlim(0.3, 2.85)
    ax.set_box_aspect((2.1, 2.9, 2.55), zoom=1.5)
    ax.view_init(elev=24, azim=-45)
    ax.set_axis_off()
    fig.savefig(output / "Camera_layout_geometry_guide.png", dpi=220, facecolor="white")
    plt.close(fig)
    (output / "Camera_layout_geometry.json").write_text(json.dumps(geometry, indent=2) + "\n")


if __name__ == "__main__":
    main()
