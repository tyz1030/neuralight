from dataclasses import dataclass
import numpy as np
import torch

@dataclass(frozen=True)
class CameraIntrinsics():
    _width: int = 3012
    _height: int = 2012
    focal_length: tuple[float, float] = 3.53992755e+03, 3.54471685e+03
    center_of_projection: tuple[float, float] = 1.50238768e+03, 9.58430303e+02
    distortion_coeff: tuple[float, float, float, float, float] = - \
        6.37672900e-02, 5.89348185e-01, -4.06256849e-03, 1.70692496e-03, -2.81961915e+00

    @property
    def camera_matrix(self)-> np.ndarray:
        return np.array([[self.focal_length[0], 0, self.center_of_projection[0]], [0, self.focal_length[1], self.center_of_projection[1]], [0, 0, 1]])

    @property
    def distortion_matrix(self)-> np.ndarray:
        return np.array([self.distortion_coeff])
    
    @property
    def width(self): return self._width

    @property
    def height(self): return self._height

    @property
    def focal_len(self): return (self.focal_length[0]+self.focal_length[1])/2