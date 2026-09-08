#!/usr/bin/env bash
set -euo pipefail

cd /home/liutong/Projects/openpi-5090
git submodule update --init third_party/libero

source /home/liutong/miniconda3/etc/profile.d/conda.sh
set +u
conda activate openpi-libero
set -u

site_packages=$(python -c 'import site; print(site.getsitepackages()[0])')
if ! grep -Fqx \
  /home/liutong/Projects/openpi-5090/packages/openpi-client/src \
  "${site_packages}/_editable_impl_openpi_client.pth" 2>/dev/null; then
  python -m pip install hatchling editables
  python -m pip install \
    --no-build-isolation \
    --no-deps \
    --editable /home/liutong/Projects/openpi-5090/packages/openpi-client
fi
export LIBERO_CONFIG_PATH=/home/liutong/.cache/openpi/libero
export PYTHONPATH=/home/liutong/Projects/openpi-5090/third_party/libero:${PYTHONPATH:-}
python - <<'PY'
import pathlib

import yaml


package_root = pathlib.Path("/home/liutong/Projects/openpi-5090/third_party/libero/libero/libero")
config_dir = pathlib.Path("/home/liutong/.cache/openpi/libero")
config_dir.mkdir(parents=True, exist_ok=True)
config = {
    "benchmark_root": str(package_root),
    "bddl_files": str(package_root / "bddl_files"),
    "init_states": str(package_root / "init_files"),
    "datasets": "/opt/liutong",
    "assets": str(package_root / "assets"),
}
config_path = config_dir / "config.yaml"
config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
print(config_path)
PY

python - <<'PY'
import imageio
import libero
from libero.libero import benchmark
import mujoco
import openpi_client
from openpi_client import websocket_client_policy
import yaml

del imageio, libero, mujoco, websocket_client_policy, yaml
print(benchmark.__file__)
print(openpi_client.__file__)
print(sorted(benchmark.get_benchmark_dict()))
PY
