# Copyright 2024 Kaining Huang and Tianyi Zhang. All rights reserved.

from dataclasses import dataclass
from .base_config import CameraIntrinsics

@dataclass(frozen=True)
class FireflyIntrinsicsFen17(CameraIntrinsics):
    _width: int = 1440
    _height: int = 1080
    focal_length: tuple[float, float] = 1135.6372, 1133.8272
    center_of_projection: tuple[float, float] = 721.376, 592.749
    distortion_coeff: tuple[float, float, float, float, float] = \
    -0.38242347,  0.20094933, -0.00268699,  0.00199554, -0.07069653
