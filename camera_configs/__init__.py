# Copyright 2024 Kaining Huang and Tianyi Zhang. All rights reserved.

from .base_config import CameraIntrinsics
from .firefly_config import FireflyIntrinsics
from .firefly_feb17_config import FireflyIntrinsicsFen17

def camera_config_factory(camera_id: str = "Firefly")->CameraIntrinsics:
    if camera_id == "Firefly":
        return FireflyIntrinsics()
    elif camera_id == "FireflyFeb17":
        return FireflyIntrinsicsFen17()
    elif camera_id == "SonyA7":
        return CameraIntrinsics()
    else:
        raise NotImplementedError(f"Camera name {camera_id} is not recognized.")