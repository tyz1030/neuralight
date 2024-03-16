import logging
from PySide6.QtCore import QThread, Signal
import torch

from config import GUIConfig, TrainingConfig
import torch
from config import TrainingConfig
from shading import ShadingModel
import torch.nn as nn
from tqdm import tqdm

class TrainingThread(QThread):
    update_images = Signal(object, object)
    update_shading_model_param = Signal(object, object, object, object, object, object, object)
    update_progress_bar = Signal(int)
    training_stopped = Signal()

    def __init__(self, shading_model: ShadingModel, training_dataloader, render, train_params, num_epochs, lr):
        super().__init__()
        self.training_config = TrainingConfig()
        self.GUI_config = GUIConfig()
        self.shading_model = shading_model
        self.training_dataloader = training_dataloader
        self.render = render
        self._is_running = True
        self.train_params = train_params
        self.num_epochs = num_epochs
        self.lr = lr

    def run(self):
        logging.basicConfig(filename='train.log', level=logging.INFO, filemode='w', format='%(name)s - %(levelname)s - %(message)s')
        for i in range(self.GUI_config.training_update_times):
            logging.info("training times: " + str(i))
            if not self._is_running:
                self.training_stopped.emit()
                return
            self.start_training(i)
            logging.info("current shading model: " + self.shading_model.light.name)
            logging.info("shading model light parameters after training: ")
            logging.info("albedo: " + str(self.shading_model.albedo))
            logging.info("ambient: " + str(self.shading_model.ambient_light))
            logging.info("gamma: " + str(self.shading_model.light.gamma))
            logging.info("tau: " + str(self.shading_model.light.tau))
            if hasattr(self.shading_model.light, 'sigma'):
                logging.info("sigma: " + str(self.shading_model.light.sigma))
            logging.info("_t_vec: " + str(self.shading_model.light._t_vec))
            logging.info("_r_l2c_SO3: " + str(self.shading_model.light._r_l2c_SO3.log()))
            imgs_raw, imgs_rendered = self.render()
            if self._is_running:
                self.update_images.emit(imgs_raw, imgs_rendered)
                if hasattr(self.shading_model.light, 'sigma'):
                    if self.shading_model.light.sigma.ndim == 0:
                        self.update_shading_model_param.emit(self.shading_model.albedo, self.shading_model.light.gamma, self.shading_model.light.tau, self.shading_model.ambient_light, self.shading_model.light._t_vec, self.shading_model.light._r_l2c_SO3.log(), [self.shading_model.light.sigma, 0])
                    else:
                        self.update_shading_model_param.emit(self.shading_model.albedo, self.shading_model.light.gamma, self.shading_model.light.tau, self.shading_model.ambient_light, self.shading_model.light._t_vec, self.shading_model.light._r_l2c_SO3.log(), [self.shading_model.light.sigma[0], self.shading_model.light.sigma[1]])
                else:
                    self.update_shading_model_param.emit(self.shading_model.albedo, self.shading_model.light.gamma, self.shading_model.light.tau, self.shading_model.ambient_light, self.shading_model.light._t_vec, self.shading_model.light._r_l2c_SO3.log(), [0, 0])
        self.training_stopped.emit()
                    
    def get_learning_param(self, checked_param):
        match checked_param:
            case "Albedo":
                return [{'params': [self.shading_model.albedo_log], 'lr': self.lr['Albedo'], "name": "albedo"}]
            case "gamma":
                return [{'params': [self.shading_model.light.gamma_log], 'lr': self.lr['gamma'], "name": "gamma"}]
            case "tau":
                return [{'params': [self.shading_model.light.tau_log], 'lr': self.lr['tau'], "name": "tau"}]
            case "Ambient":
                return [{'params': [self.shading_model.ambient_light_log], 'lr': self.lr['Ambient'], "name": "ambient"}]
            case "Rotation":
                return [{'params': [self.shading_model.light._r_l2c_SO3], 'lr': self.lr['Rotation'], "name": "r_vec"}]
            case "Translation":
                return [{'params': [self.shading_model.light._t_vec], 'lr': self.lr['Translation'], "name": "t_vec"}]
            case "\u03C3_x":
                return [{'params': [self.shading_model.light.sigma], 'lr': self.lr['\u03C3_x'], 'name': 'sigma'}]
            case "MLP parameters":
                return [{'params': self.shading_model.light.mlp.parameters(), 'lr': self.lr['MLP parameters'], 'name': 'mlp0'}]
            case _:
                return []


    def start_training(self, round: int):       
        num_epoch = (int)(self.num_epochs / self.GUI_config.training_update_times)

        loss_fn = nn.L1Loss()
        l = []
        for train_param in self.train_params:
            l += self.get_learning_param(train_param)
        optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.shading_model = self.shading_model.to(self.training_config.device)
        for epoch in tqdm(range(num_epoch)):
            if not self._is_running:
                self.training_stopped.emit()
                return
            self.update_progress_bar.emit(round * (int)(self.num_epochs / self.GUI_config.training_update_times) + epoch)
            for itr, (pts, intensities, rvec_w2c, tvec_w2c, _, _) in enumerate(self.training_dataloader):
                rendered_intensities = self.shading_model(pts, rvec_w2c, tvec_w2c)
                loss = loss_fn(rendered_intensities, intensities)
                loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    def stop(self):
        self._is_running = False