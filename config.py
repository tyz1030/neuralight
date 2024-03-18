# Copyright 2024 Kaining Huang and Tianyi Zhang. All rights reserved.

from dataclasses import dataclass, field
import torch
from typing import Optional
import numpy as np
from typing import Dict


@dataclass(frozen=True)
class TrainingConfig():
    lr :        float = 1e-2
    epoch:      int   = 1000
    batch_size: int   = 1
    brdf:       str   = "Lambertian" # "Lambertian" or "Microfacet-Mono", "DisneyDiffuse"
    light:      str   = "Gaussian1D" # "Gaussian2D", "Gaussian1D" or "PointLightSource"
    image_path: str   = "toy_data_firefly"
    camera_name: str  = "FireflyFeb17"
    device:     str   = "cuda:0" # cuda:0 or cpu
    unit:       str   = "mm" # "mm" or "inch"
    albedo_lr:  float = 0.1
    t_vec_lr:  float = 1e-3
    r_vec_lr:   float = 1e-3
    ambient_light_lr: float = 1e-2
    sigma_lr:   float = 1.
    gamma_lr:   float = 1e-2
    tau_lr:     float = 1e-2
    undistort_image: bool = False
    lightmodel_lr: float = 5e-3 # lr for e.g. Sigma of Gaussians, Weights of MLP
    split: int = 4

@dataclass(frozen=True)
class GUIConfig():
    albedo_step: float = 0.1
    albedo_min:  float = 0.
    albedo_max:  float = 1000.
    albedo_default: float = 20.
    albedo_decimal: int = 1
    gamma_step: float = 0.1
    gamma_min: float = -99.
    gamma_max: float = 99.
    gamma_default: float = 1.
    gamma_decimal: int = 2
    tau_step: float = 0.1
    tau_min: float = -99.
    tau_max: float = 99.
    tau_default: float = 0.5
    tau_decimal: int = 2
    ambient_step: float = 0.1
    ambient_min: float = 0.
    ambient_step: float = 0.1
    ambient_min: float = 0.
    ambient_max: float = 10.
    ambient_default: float = 0.5
    ambient_decimal: int = 3
    r_layout_step: float = 0.001
    r_layout_min: float = -10.
    r_layout_max:  float = 10.
    r_layout_default: Dict[str, float] = field(default_factory = lambda: ({"x": -0.102, "y": -0.372, "z": 0.193}))
    r_layout_decimal: int = 3
    t_layout_step: float = 0.001
    t_layout_min: float = -10.
    t_layout_max:  float = 10.
    t_layout_default: Dict[str, float] = field(default_factory = lambda: ({"x": 0.3, "y": -0.000, "z": 0.000}))
    t_layout_decimal: int = 3
    sigma_step: float = 0.1
    sigma_min: float = -100.
    sigma_max: float = 100.
    sigma_default: float = 11.
    sigma_decimal: int = 1
    dataset_default: str = "Training set"
    page_step: int = 1 
    page_min: int = 1
    page_max: int = 9
    page_default: int = 1
    zoom_step: float = 0.1
    zoom_min: float = 0.1
    zoom_max: float = 10.
    zoom_default: float = 1.1
    num_epoch_step: int = 1
    num_epoch_max: int = 2**31 - 1
    light_source_default: str = "1D MLP" # "Gaussian2D", "Gaussian1D", "PointLightSource", "1D MLP", "2D MLP"
    training_update_times: int = 100
    error_decimal: int = 3
    lr_input_width: int = 70
    progress_bar_length: int = 120



@dataclass(frozen=True)
class CamLightCalibTarget():
    '''
    The physical information about the calibration target used to calibrate the camera and light system
    default tag size is 2 inch and paper is 11*17 inch
    '''
    tag2tag_x: int = 15
    tag2tag_y: int = 9
    spacer_size: float = 2.0/6
    tag_size: float = 8.0/6
    tag0_corners: torch.Tensor = torch.tensor([[spacer_size, spacer_size+tag_size, 0],
                                               [spacer_size+tag_size, spacer_size+tag_size, 0], 
                                               [spacer_size+tag_size, spacer_size, 0], 
                                               [spacer_size, spacer_size, 0]])
    _tag_corners: torch.Tensor = torch.stack((tag0_corners,
                                             tag0_corners+torch.tensor([[tag2tag_x, 0, 0]]),
                                             tag0_corners+torch.tensor([[tag2tag_x, tag2tag_y, 0]]),
                                             tag0_corners+torch.tensor([[0, tag2tag_y, 0]])),
                                             dim=0)

    def get_tag_corners(self, unit: Optional[str] = 'inch'):
        if unit == 'inch':
            return self._tag_corners
        elif unit == 'mm':
            return 25.4*self._tag_corners
        elif unit == 'm':
            return 0.0254*self._tag_corners
        else:
            return NotImplementedError("This Type of unit is not implemented.")



@dataclass(frozen=True)
class CamLightCalibTargetWall():
    '''
    The physical information about the calibration target used to calibrate the camera and light system
    default tag size is 2 inch and paper is 11*17 inch
    '''
    tag2tag_x: int = 48 #inch
    tag2tag_y: int = 28
    spacer_size: float = 4.0/6.0
    tag_size: float = 16.0/6.0
    tag0_corners: torch.Tensor = torch.tensor([[spacer_size, spacer_size+tag_size, 0],
                                               [spacer_size+tag_size, spacer_size+tag_size, 0], 
                                               [spacer_size+tag_size, spacer_size, 0], 
                                               [spacer_size, spacer_size, 0]])
    _tag_corners: torch.Tensor = torch.stack((tag0_corners,
                                             tag0_corners+torch.tensor([[tag2tag_x, 0, 0]]),
                                             tag0_corners+torch.tensor([[tag2tag_x, tag2tag_y, 0]]),
                                             tag0_corners+torch.tensor([[0, tag2tag_y, 0]])),
                                             dim=0)

    def get_tag_corners(self, unit: Optional[str] = 'inch'):
        if unit == 'inch':
            return self._tag_corners
        elif unit == 'mm':
            return 25.4*self._tag_corners
        elif unit == 'm':
            return 0.0254*self._tag_corners
        else:
            return NotImplementedError("This Type of unit is not implemented.")
        

# test code
if __name__ == "__main__":
    target = CamLightCalibTarget()
    print(target.get_tag_corners())    
    print(target.get_tag_corners(unit = 'mm'))    
    print(target.get_tag_corners(unit = 'cm'))