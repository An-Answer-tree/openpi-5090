import dataclasses
import types

import jax
import pytest
import torch

from openpi.models import pi0_config
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


class _IndexDataset:
    """Small dataset that exposes the sampled indices."""

    def __init__(self, size: int):
        self._size = size
        self.num_reads = 0

    def __getitem__(self, index: int) -> torch.Tensor:
        self.num_reads += 1
        return torch.tensor(index)

    def __len__(self) -> int:
        return self._size


def test_torch_data_loader():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 16)

    loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=4,
        num_batches=2,
    )
    batches = list(loader)

    assert len(batches) == 2
    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_torch_data_loader_infinite():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 4)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4)
    data_iter = iter(loader)

    for _ in range(10):
        _ = next(data_iter)


def test_create_torch_dataset_uses_global_episode_indices(monkeypatch):
    class FakeLeRobotDataset:
        def __init__(self, repo_id, *, delta_timestamps):
            del repo_id, delta_timestamps
            self.episode_data_index = {
                "from": torch.tensor([0, 2, 5]),
                "to": torch.tensor([2, 5, 6]),
            }

        def __getitem__(self, index):
            return index

        def __len__(self):
            return 6

    monkeypatch.setattr(
        _data_loader.lerobot_dataset,
        "LeRobotDatasetMetadata",
        lambda repo_id: types.SimpleNamespace(fps=10),
    )
    monkeypatch.setattr(_data_loader.lerobot_dataset, "LeRobotDataset", FakeLeRobotDataset)

    dataset = _data_loader.create_torch_dataset(
        _config.DataConfig(repo_id="test"),
        action_horizon=2,
        model_config=object(),
        episodes=[2, 0],
    )

    assert [dataset[index] for index in range(len(dataset))] == [5, 0, 1]


def test_torch_data_loader_parallel():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 10)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4, num_batches=2, num_workers=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


@pytest.mark.parametrize("num_workers", [0, 2])
def test_torch_data_loader_resumes_shuffle_order(num_workers: int):
    dataset = _IndexDataset(11)
    loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=4,
        shuffle=True,
        num_batches=9,
        num_workers=num_workers,
        seed=42,
        framework="pytorch",
    )
    expected = [batch.tolist() for batch in loader]

    resumed_dataset = _IndexDataset(11)
    resumed_loader = _data_loader.TorchDataLoader(
        resumed_dataset,
        local_batch_size=4,
        shuffle=True,
        num_batches=4,
        num_workers=num_workers,
        seed=42,
        framework="pytorch",
        start_batch=5,
    )
    actual = [batch.tolist() for batch in resumed_loader]

    assert actual == expected[5:9]


def test_torch_data_loader_resume_does_not_read_skipped_samples():
    dataset = _IndexDataset(11)
    loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=4,
        shuffle=True,
        num_batches=1,
        seed=42,
        framework="pytorch",
        start_batch=5,
    )

    list(loader)

    assert dataset.num_reads == 4


def test_with_fake_dataset():
    config = _config.get_config("debug")

    loader = _data_loader.create_data_loader(config, skip_norm_stats=True, num_batches=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == config.batch_size for x in jax.tree.leaves(batch))

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


def test_with_real_dataset():
    config = _config.get_config("pi0_aloha_sim")
    config = dataclasses.replace(config, batch_size=4)

    loader = _data_loader.create_data_loader(
        config,
        # Skip since we may not have the data available.
        skip_norm_stats=True,
        num_batches=2,
        shuffle=True,
    )
    # Make sure that we can get the data config.
    assert loader.data_config().repo_id == config.data.repo_id

    batches = list(loader)

    assert len(batches) == 2

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)
