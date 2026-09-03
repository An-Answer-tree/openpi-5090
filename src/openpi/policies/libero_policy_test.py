import numpy as np

from openpi.models import model as _model
from openpi.policies import libero_policy


def test_libero_inputs_without_wrist_image():
    base_image = np.ones((224, 224, 3), dtype=np.uint8)
    transform = libero_policy.LiberoInputs(model_type=_model.ModelType.PI05, use_wrist_image=False)

    result = transform(
        {
            "observation/image": base_image,
            "observation/state": np.zeros(8, dtype=np.float32),
        }
    )

    np.testing.assert_array_equal(result["image"]["base_0_rgb"], base_image)
    np.testing.assert_array_equal(result["image"]["left_wrist_0_rgb"], np.zeros_like(base_image))
    assert not result["image_mask"]["left_wrist_0_rgb"]
