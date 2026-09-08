import collections
import dataclasses
import logging
import math
import pathlib

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy
import tqdm

from examples.libero.multiview_eval.tuned_cameras import TunedCameraInstaller


@dataclasses.dataclass
class EvalArgs:
    """Arguments for fixed-camera LIBERO evaluation."""

    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5
    base_image_key: str = "agentview_image"
    task_suite_name: str = "libero_spatial"
    num_steps_wait: int = 10
    num_trials_per_task: int = 50
    video_out_path: str = "data/libero/videos"
    save_videos: bool = False
    seed: int = 7


@dataclasses.dataclass(frozen=True)
class ViewSpec:
    camera_name: str
    observation_key: str


libero_env_resolution = 256
libero_dummy_action = [0.0] * 6 + [-1.0]
task_suite_max_steps = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
}
image_key_to_camera_name = {
    "agentview_image": "agentview",
    "topview_image": "operation_topview",
    "leftview_image": "operation_leftview",
    "rightview_image": "operation_rightview",
    "backview_image": "operation_backview",
}


def resolve_view(base_image_key: str) -> ViewSpec:
    """Resolve a LeRobot image key to its LIBERO camera observation."""

    camera_name = image_key_to_camera_name[base_image_key]
    return ViewSpec(camera_name=camera_name, observation_key=f"{camera_name}_image")


def quat_to_axis_angle(quaternion: np.ndarray) -> np.ndarray:
    scalar = np.clip(quaternion[3], -1.0, 1.0)
    denominator = np.sqrt(1.0 - scalar * scalar)
    if math.isclose(denominator, 0.0):
        return np.zeros(3)
    return quaternion[:3] * 2.0 * math.acos(scalar) / denominator


class ObservationPreprocessor:
    """Convert a LIBERO observation to the OpenPI policy input format."""

    def __init__(self, view: ViewSpec, resize_size: int) -> None:
        self._view = view
        self._resize_size = resize_size

    def prepare(self, observation: dict, prompt: str) -> tuple[dict, np.ndarray]:
        base_image = self._prepare_image(observation[self._view.observation_key])
        wrist_image = self._prepare_image(observation["robot0_eye_in_hand_image"])
        policy_input = {
            "observation/image": base_image,
            "observation/wrist_image": wrist_image,
            "observation/state": np.concatenate(
                (
                    observation["robot0_eef_pos"],
                    quat_to_axis_angle(observation["robot0_eef_quat"]),
                    observation["robot0_gripper_qpos"],
                )
            ),
            "prompt": str(prompt),
        }
        return policy_input, base_image

    def _prepare_image(self, image: np.ndarray) -> np.ndarray:
        image = np.ascontiguousarray(image[::-1])
        return image_tools.convert_to_uint8(image_tools.resize_with_pad(image, self._resize_size, self._resize_size))


class MultiviewLiberoEvaluator:
    """Evaluate an OpenPI policy with the fixed-dataset camera profile."""

    def __init__(self, args: EvalArgs) -> None:
        self._args = args
        self._view = resolve_view(args.base_image_key)
        self._max_steps = task_suite_max_steps[args.task_suite_name]
        self._client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port, ping_interval=None)
        self._preprocessor = ObservationPreprocessor(self._view, args.resize_size)

    def run(self) -> None:
        np.random.seed(self._args.seed)
        TunedCameraInstaller.install()
        task_suite = benchmark.get_benchmark_dict()[self._args.task_suite_name]()

        total_episodes = 0
        total_successes = 0
        logging.info("Task suite: %s", self._args.task_suite_name)
        for task_id in tqdm.tqdm(range(task_suite.n_tasks)):
            task_episodes, task_successes = self._run_task(task_suite, task_id)
            total_episodes += task_episodes
            total_successes += task_successes
            logging.info(
                "Task %d success rate: %.4f (%d/%d)",
                task_id,
                task_successes / task_episodes,
                task_successes,
                task_episodes,
            )
            logging.info(
                "Running success rate: %.4f (%d/%d)",
                total_successes / total_episodes,
                total_successes,
                total_episodes,
            )

        logging.info(
            "Final success rate: %.4f (%d/%d)",
            total_successes / total_episodes,
            total_successes,
            total_episodes,
        )

    def _run_task(self, task_suite, task_id: int) -> tuple[int, int]:
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        environment, task_description = self._create_environment(task)
        successes = 0
        try:
            for episode_index in tqdm.tqdm(range(self._args.num_trials_per_task)):
                successes += self._run_episode(
                    environment,
                    initial_states[episode_index],
                    task_description,
                    episode_index,
                )
        finally:
            environment.close()
        return self._args.num_trials_per_task, successes

    def _create_environment(self, task):
        task_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        camera_names = list(dict.fromkeys((self._view.camera_name, "robot0_eye_in_hand")))
        environment = OffScreenRenderEnv(
            bddl_file_name=task_file,
            camera_heights=libero_env_resolution,
            camera_widths=libero_env_resolution,
            camera_names=camera_names,
        )
        environment.seed(self._args.seed)
        return environment, task.language

    def _run_episode(self, environment, initial_state, task_description: str, episode_index: int) -> bool:
        logging.info("Task: %s; episode: %d", task_description, episode_index)
        environment.reset()
        observation = environment.set_init_state(initial_state)
        action_plan = collections.deque()
        replay_images = []
        success = False

        for step in range(self._max_steps + self._args.num_steps_wait):
            if step < self._args.num_steps_wait:
                observation, _, _, _ = environment.step(libero_dummy_action)
                continue

            policy_input, replay_image = self._preprocessor.prepare(observation, task_description)
            if self._args.save_videos:
                replay_images.append(replay_image)
            if not action_plan:
                actions = self._client.infer(policy_input)["actions"]
                if len(actions) < self._args.replan_steps:
                    raise ValueError(
                        f"Policy returned {len(actions)} actions, fewer than replan_steps={self._args.replan_steps}."
                    )
                action_plan.extend(actions[: self._args.replan_steps])

            observation, _, done, _ = environment.step(action_plan.popleft().tolist())
            if done:
                success = True
                break

        if self._args.save_videos:
            self._save_replay(task_description, episode_index, replay_images, success=success)
        logging.info("Success: %s", success)
        return success

    def _save_replay(
        self,
        task_description: str,
        episode_index: int,
        replay_images: list[np.ndarray],
        *,
        success: bool,
    ) -> None:
        output_dir = pathlib.Path(self._args.video_out_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        task_name = task_description.replace(" ", "_")
        result = "success" if success else "failure"
        imageio.mimwrite(
            output_dir / f"{task_name}_{episode_index:02d}_{result}.mp4",
            replay_images,
            fps=10,
        )
