# Copyright 2024 Kaining Huang and Tianyi Zhang. All rights reserved.

from torch.utils.data import Dataset
import os
import numpy as np
import cv2 as cv
from config import CamLightCalibTargetWall
from camera_configs import camera_config_factory, CameraIntrinsics
from dt_apriltags import Detector
import torch
from typing import Tuple, List
from utils import find_mask
import matplotlib.pyplot as plt
from torch import Tensor
from lietorch import SO3
import torch.nn.functional as F

class CaliDataset(Dataset):
    """Camera and Light calibration"""

    def __init__(
        self, image_path: str, device: str = "cuda:0", undistort_imgs: bool = True, camera_name: str = "Firefly"
    ) -> None:
        """
        Arguments:
        """
        self.device = device
        self.camera_name = camera_name
        files = os.listdir(image_path)
        image_extensions = [".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff"]
        image_files = [
            f for f in files if any(f.endswith(ext) for ext in image_extensions)
        ]
        image_files = sorted(image_files)
        if undistort_imgs:
            imgs = [
                cv.imread(os.path.join(image_path, image_file), cv.IMREAD_GRAYSCALE)
                for image_file in image_files
            ]
            images, self.new_cam_intrin = self.undistort_images(imgs)
        else:
            images = [
                cv.imread(os.path.join(image_path, image_file), cv.IMREAD_GRAYSCALE) for image_file in image_files
            ]
            cam_intri: CameraIntrinsics = camera_config_factory('FireflyFeb17')
            mtx = cam_intri.camera_matrix
            dist = cam_intri.distortion_matrix
            w, h = cam_intri.width, cam_intri.height
            self.new_cam_intrin, _ = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 0, (w,h), centerPrincipalPoint=True)

        self.target = CamLightCalibTargetWall()
        self.at_detector = self.get_apriltag_detector()
        
        self.pts, self.intensities, self.cam_poses_rvec, self.cam_poses_tvec, self.images_list, self.masks = self.get_3d_pts(images)
        

    def __len__(self) -> int:
        return len(self.pts)

    def __getitem__(self, i:int)->Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        return self.pts[i], self.intensities[i], self.cam_poses_rvec[i], self.cam_poses_tvec[i], self.images_list[i], self.masks[i]

    def get_apriltag_detector(self) -> Detector:
        return Detector(
            searchpath=["apriltags"],
            families="tag36h11",
            nthreads=1,
            quad_decimate=3.0,
            quad_sigma=0.0,
            refine_edges=2,
            decode_sharpening=0.5,
            debug=0,
        )

    def get_grid_xyz(self, h, w) -> Tensor:
        grid_x, grid_y = torch.meshgrid(
            torch.arange(w), torch.arange(h), indexing="xy",
        )
        grid_x_centered, grid_y_centered = (
            grid_x + 0.5 - w / 2.0,
            grid_y + 0.5 - h / 2.0,
        )
        grid_z = torch.ones_like(grid_x) * self.new_cam_intrin[0, 0]
        grid_xyz = torch.stack([grid_x_centered, grid_y_centered, grid_z], dim=-1).to(torch.double)
        return grid_xyz
    

    def corners_to_3d_pts(self, target_corners: Tensor, tag_corners: Tensor, masked_xyz_list: Tensor)->Tensor:
        tag_corners = tag_corners.view(-1, 2)
        retval, r_vec, t_vec = cv.solvePnP(
            target_corners.numpy(), tag_corners.numpy(), self.new_cam_intrin, None
        )
        r_vec_w2c = torch.tensor(r_vec, device=self.device).squeeze().unsqueeze(0)
        t_w2c = torch.tensor(t_vec, device=self.device).squeeze().unsqueeze(0)
        R_w2c = SO3.exp(r_vec_w2c)
        R_c2w = R_w2c.inv()
        t_c2w = -R_c2w.act(t_w2c)
        origin = t_c2w.clone()
        w = R_c2w.act(masked_xyz_list.transpose(0,1))#.double())
        w = F.normalize(w, dim=-1)
        # We want to find the intersection of each ray and z=0 plane. r = o + wt
        t = -origin[:, -1]/w[:, -1]
        r = origin+w*t[:, None]
        return r, r_vec_w2c, t_w2c

    def get_3d_pts(self, images:np.ndarray) -> Tuple[List[Tensor], List[Tensor], List[Tensor], List[Tensor], List[Tensor], Tensor]:
        pts = []
        intensities = []
        cam_poses_rvec = []
        cam_poses_tvec = []
        centers = []
        tags_corners = []
        imgs = []
        h, w = images[0].shape
        grid_xyz = self.get_grid_xyz(h,w)
        grid_xyz_list = grid_xyz.view(-1, 3).to(self.device)


        target_corners = self.target.get_tag_corners(unit="m").view(-1, 3)
        
        for image in images:
            # curve image values for better detection performance
            img_curved = np.clip(np.power(image / 255.0, 0.45), 0.0, 1.0)
            img_curved = (img_curved * 255).astype(np.uint8)
            tags = self.at_detector.detect(img_curved)
            centers_curr_img = []
            tag_corners_curr_img = []
            img_viz = img_curved.copy()
            for tag in tags:
                centers_curr_img.append(tag.center)
                tag_corners_curr_img.append(tag.corners)
                img_viz = cv.circle(img_viz, (int(tag.center[0]), int(tag.center[1])), 6, [255, 0, 255])
                # break
                for corner in tag.corners:
                    img_viz = cv.circle(img_viz, (int(corner[0]), int(corner[1])), 3, [0, 0, 255])
            if len(centers_curr_img) == 4:
                centers.append(centers_curr_img)
                tags_corners.append(tag_corners_curr_img)
                imgs.append(torch.tensor(image, device=self.device))
                
            # cv.imshow("asd", (img_viz))
            # cv.waitKey(0)
            # cv.destroyAllWindows()

        tags_corners = torch.tensor(np.asarray(tags_corners))
        centers = torch.tensor(np.asarray(centers))
        masks = find_mask(centers, image.shape[0:2], visualize=False, scale=0.85)
        masks = masks.to(self.device)

        for tag_corners, mask in zip(tags_corners, masks):
            masked_xyz_list = grid_xyz_list[mask.view(-1)].transpose(0, 1)
            r, r_vec, t_vec = self.corners_to_3d_pts(target_corners , tag_corners, masked_xyz_list)
            cam_poses_rvec.append(r_vec)
            cam_poses_tvec.append(t_vec)
            # plt.scatter(r[0], r[1], s=0.01)
            # plt.axis('equal')
            # plt.show()
            pts.append(r)

        for img, mask in zip(images, masks):
            intensity = torch.tensor(img, device=self.device, dtype=torch.float64).view(-1)[mask.view(-1)]
            intensities.append(intensity)

        # self.visualize_image_w_mask(imgs, masks)
            
        return pts, intensities, cam_poses_rvec, cam_poses_tvec, imgs, masks

    def visualize_image_w_mask(self, imgs: np.ndarray, masks: Tensor)->None:
        for mask, img in zip(masks, imgs):
            img_overlap = (0.2 * img + 0.8 * img * mask) / 255.0
            cv.imshow("ads", cv.pyrDown(img_overlap.cpu().numpy()))
            cv.waitKey(0)
            cv.destroyAllWindows()

    def undistort_images(self, images: np.ndarray | List[np.ndarray]) -> np.ndarray | List[np.ndarray]:
        cam_intri: CameraIntrinsics = camera_config_factory(self.camera_name)
        mtx = cam_intri.camera_matrix
        dist = cam_intri.distortion_matrix
        w, h = cam_intri.width, cam_intri.height
        new_cam_intrin, _ = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 0, (w,h), centerPrincipalPoint=True)
        if type(images) is list:
            undistorted_imgs = [cv.undistort(img, mtx, dist, None, new_cam_intrin) for img in images]
        else:
            raise NotImplementedError("Not Implemented")
        return undistorted_imgs, new_cam_intrin
