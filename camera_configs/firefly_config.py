# Copyright 2024 Kaining Huang and Tianyi Zhang. All rights reserved.

from dataclasses import dataclass
from .base_config import CameraIntrinsics

@dataclass(frozen=True)
class FireflyIntrinsics(CameraIntrinsics):
    _width: int = 1440
    _height: int = 1080
    focal_length: tuple[float, float] = 1.77956506e+03, 1.78166582e+03
    center_of_projection: tuple[float, float] = 6.73645910e+02, 5.63353050e+02
    distortion_coeff: tuple[float, float, float, float, float] = \
    -0.35078876,  0.07201943, -0.00047334,  0.00059974,  0.25438148