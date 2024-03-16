# Copyright 2024 Kaining Huang and Tianyi Zhang. All rights reserved.

import string
import sys
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QLabel,
    QSpinBox,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QSizePolicy,
    QFrame,
)
from PySide6.QtGui import QPixmap, QImage, QFont
from PySide6.QtCore import Qt

import torch
from config import TrainingConfig, GUIConfig
from shading import ShadingModel
from cali_dataset import CaliDataset
from torch.utils.data import DataLoader, random_split
import cv2 as cv
import numpy as np
import os

from train_thread import TrainingThread
from widget_creator import WidgetFactory, set_button_style


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.training_config = TrainingConfig()
        self.GUI_config = GUIConfig()
        self.shading_model = ShadingModel(
            brdf=self.training_config.brdf,
            light=self.training_config.light,
            device=self.training_config.device,
        )
        self.dataset = CaliDataset(
            image_path=self.training_config.image_path,
            device=self.training_config.device,
            undistort_imgs=self.training_config.undistort_image,
            camera_name=self.training_config.camera_name
        )

        test_size = int(len(self.dataset) / self.training_config.split)
        train_size = len(self.dataset) - test_size

        self.training_dataset, self.testing_dataset = random_split(self.dataset, [train_size, test_size])

        self.training_dataloader = DataLoader(
            self.training_dataset, batch_size=self.training_config.batch_size
        )
        self.testing_dataloader = DataLoader(
            self.testing_dataset, batch_size=self.training_config.batch_size
        )

        self.current_dataset = self.training_dataset
        self.current_dataloader = self.training_dataloader

        self.shading_model.to(device=self.training_config.device)

        self.training_thread = None

        # Main widget and layout
        main_widget = QWidget(self)
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # Font (bold)
        font = QFont()
        font.setBold(True)

        # Image label
        self.image_layout = QHBoxLayout()
        raw_image_layout = QVBoxLayout()
        label = QLabel("Raw Image")
        label.setFont(font)
        label.setAlignment(Qt.AlignCenter)
        raw_image_layout.addWidget(label)
        self.image_label_raw = WidgetFactory.get_widget("image_label")
        raw_image_layout.addWidget(self.image_label_raw)
        raw_image_layout.setAlignment(Qt.AlignCenter)
        self.image_layout.addLayout(raw_image_layout)

        rendered_image_layout = QVBoxLayout()
        label = QLabel("Rendered Image")
        label.setFont(font)
        label.setAlignment(Qt.AlignCenter)
        rendered_image_layout.addWidget(label)
        self.image_label_rendered = WidgetFactory.get_widget("image_label")
        rendered_image_layout.addWidget(self.image_label_rendered)
        rendered_image_layout.setAlignment(Qt.AlignCenter)
        self.image_layout.addLayout(rendered_image_layout)

        diff_image_layout = QVBoxLayout()
        label = QLabel("Difference Image")
        label.setFont(font)
        label.setAlignment(Qt.AlignCenter)
        diff_image_layout.addWidget(label)
        self.image_label_diff = WidgetFactory.get_widget("image_label")
        diff_image_layout.addWidget(self.image_label_diff)
        diff_image_layout.setAlignment(Qt.AlignCenter)
        self.image_layout.addLayout(diff_image_layout)

        layout.addLayout(self.image_layout)

        # Parameters layout
        self.param_layout = QHBoxLayout()
        self.checkbox_label_pairs = []

        # Optical parameters frame
        optical_frame = QFrame()
        optical_frame.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        optical_layout = QVBoxLayout()
        label = QLabel("Optical Parameters")
        label.setFont(font)
        optical_layout.addWidget(label, alignment=Qt.AlignTop)
        optical_params_layout = QHBoxLayout()
        first_column_layout = QVBoxLayout()

        # Albedo
        albedo_layout = QVBoxLayout()
        hbox = QHBoxLayout()
        checkbox = WidgetFactory.get_widget("checkbox", window=self)
        label = QLabel("Albedo", self)
        self.albedo_spin_box = WidgetFactory.get_widget(
            "double_spin_box",
            window=self,
            name="albedo_spin_box",
            step=self.GUI_config.albedo_step,
            min=self.GUI_config.albedo_min,
            max=self.GUI_config.albedo_max,
            value=self.GUI_config.albedo_default,
            decimals=self.GUI_config.albedo_decimal,
        )
        hbox.addWidget(checkbox)
        hbox.addWidget(label)
        hbox.addWidget(self.albedo_spin_box, alignment=Qt.AlignRight)
        albedo_layout.addLayout(hbox)
        self.checkbox_label_pairs.append((checkbox, label))

        # Albedo learning rate
        lr_layout = QHBoxLayout()
        lr_layout.addSpacing(20)
        label = QLabel("lr")
        lr_layout.addSpacing(35)
        label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        lr_layout.addWidget(label)
        self.albedo_lr_input = WidgetFactory.get_widget(
            "lr_text_input", value=str(self.training_config.albedo_lr)
        )
        lr_layout.addWidget(self.albedo_lr_input, alignment=Qt.AlignRight)
        albedo_layout.addLayout(lr_layout)
        first_column_layout.addLayout(albedo_layout)

        # Ambient light
        ambient_layout = QVBoxLayout()
        hbox = QHBoxLayout()
        checkbox = WidgetFactory.get_widget("checkbox")
        label = QLabel("Ambient", self)
        self.ambient_spin_box = WidgetFactory.get_widget(
            "double_spin_box",
            window=self,
            name="ambient_spin_box",
            step=self.GUI_config.ambient_step,
            min=self.GUI_config.ambient_min,
            max=self.GUI_config.ambient_max,
            value=self.GUI_config.ambient_default,
            decimals=self.GUI_config.ambient_decimal,
        )
        hbox.addWidget(checkbox)
        hbox.addWidget(label)
        hbox.addWidget(self.ambient_spin_box, alignment=Qt.AlignRight)
        ambient_layout.addLayout(hbox)
        self.checkbox_label_pairs.append((checkbox, label))

        # Ambient learning rate
        lr_layout = QHBoxLayout()
        lr_layout.addSpacing(20)
        label = QLabel("lr")
        lr_layout.addSpacing(35)
        label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        lr_layout.addWidget(label)
        self.ambient_lr_input = WidgetFactory.get_widget(
            "lr_text_input", value=str(self.training_config.ambient_light_lr)
        )
        lr_layout.addWidget(self.ambient_lr_input, alignment=Qt.AlignRight)
        ambient_layout.addLayout(lr_layout)
        first_column_layout.addLayout(ambient_layout)

        optical_params_layout.addLayout(first_column_layout)
        optical_params_layout.addSpacing(10)

        falloff_column_layout = QVBoxLayout()

        # Gamma
        gamma_layout = QVBoxLayout()
        hbox = QHBoxLayout()
        checkbox = WidgetFactory.get_widget("checkbox", window=self)
        checkbox.setChecked(False)
        label = QLabel("gamma", self)
        self.gamma_spin_box = WidgetFactory.get_widget(
            "double_spin_box",
            window=self,
            name="gamma_spin_box",
            step=self.GUI_config.gamma_step,
            min=self.GUI_config.gamma_min,
            max=self.GUI_config.gamma_max,
            value=self.GUI_config.gamma_default,
            decimals=self.GUI_config.gamma_decimal,
        )
        hbox.addWidget(checkbox)
        hbox.addWidget(label)
        hbox.addWidget(self.gamma_spin_box, alignment=Qt.AlignRight)
        gamma_layout.addLayout(hbox)
        self.checkbox_label_pairs.append((checkbox, label))

        # Gamma learning rate
        lr_layout = QHBoxLayout()
        lr_layout.addSpacing(30)
        label = QLabel("lr")
        lr_layout.addSpacing(15)
        label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        lr_layout.addWidget(label)
        self.gamma_lr_input = WidgetFactory.get_widget(
            "lr_text_input", value=str(self.training_config.gamma_lr)
        )
        lr_layout.addWidget(self.gamma_lr_input, alignment=Qt.AlignRight)
        gamma_layout.addLayout(lr_layout)
        falloff_column_layout.addLayout(gamma_layout)
        
        # Tau
        tau_layout = QVBoxLayout()
        hbox = QHBoxLayout()
        checkbox = WidgetFactory.get_widget("checkbox", window=self)
        label = QLabel("tau", self)
        self.tau_spin_box = WidgetFactory.get_widget(
            "double_spin_box",
            window=self,
            name="tau_spin_box",
            step=self.GUI_config.tau_step,
            min=self.GUI_config.tau_min,
            max=self.GUI_config.tau_max,
            value=self.GUI_config.tau_default,
            decimals=self.GUI_config.tau_decimal,
        )
        hbox.addWidget(checkbox)
        hbox.addWidget(label)
        hbox.addWidget(self.tau_spin_box, alignment=Qt.AlignRight)
        tau_layout.addLayout(hbox)
        self.checkbox_label_pairs.append((checkbox, label))

        # Tau learning rate
        lr_layout = QHBoxLayout()
        lr_layout.addSpacing(30)
        label = QLabel("lr")
        lr_layout.addSpacing(15)
        label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        lr_layout.addWidget(label)
        self.tau_lr_input = WidgetFactory.get_widget(
            "lr_text_input", value=str(self.training_config.tau_lr)
        )
        lr_layout.addWidget(self.tau_lr_input, alignment=Qt.AlignRight)
        tau_layout.addLayout(lr_layout)
        falloff_column_layout.addLayout(tau_layout)

        optical_params_layout.addLayout(falloff_column_layout)
        optical_layout.addLayout(optical_params_layout)
        optical_frame.setLayout(optical_layout)
        self.param_layout.addWidget(optical_frame)
        self.param_layout.addSpacing(10)

        # Pose parameters frame
        pose_frame = QFrame()
        pose_frame.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        pose_layout = QVBoxLayout()
        label = QLabel("Pose Parameters")
        label.setFont(font)
        pose_layout.addWidget(label, alignment=Qt.AlignTop)
        pose_layout.addSpacing(6)

        pose_params_layout = QHBoxLayout()

        # Spin boxes for rotation and translation
        self.r_l2c_spin_boxes = []
        self.t_l2c_spin_boxes = []

        self.r_layout = QVBoxLayout()
        self.t_layout = QVBoxLayout()
        hbox = QHBoxLayout()
        checkbox = WidgetFactory.get_widget("checkbox", window=self)
        label = QLabel("Rotation", self)
        hbox.addWidget(checkbox)
        hbox.addWidget(label)
        self.r_layout.addLayout(hbox)
        self.checkbox_label_pairs.append((checkbox, label))
        for axis in ["x", "y", "z"]:
            sub_layout = QHBoxLayout()
            label = QLabel("r_" + axis)
            label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            sub_layout.addWidget(label)
            spin_box = WidgetFactory.get_widget(
                "double_spin_box",
                window=self,
                name="r_spin_box_" + axis,
                step=self.GUI_config.r_layout_step,
                min=self.GUI_config.r_layout_min,
                max=self.GUI_config.r_layout_max,
                value=self.GUI_config.r_layout_default[axis],
                decimals=self.GUI_config.r_layout_decimal,
            )
            sub_layout.addWidget(spin_box)
            self.r_layout.addLayout(sub_layout)
            self.r_l2c_spin_boxes.append(spin_box)

        # Rotation learning rate
        lr_layout = QHBoxLayout()
        label = QLabel("lr")
        label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        lr_layout.addWidget(label)
        self.r_lr_input = WidgetFactory.get_widget(
            "lr_text_input", value=str(self.training_config.r_vec_lr)
        )
        lr_layout.addWidget(self.r_lr_input)
        self.r_layout.addLayout(lr_layout)
        pose_params_layout.addLayout(self.r_layout)
        pose_params_layout.addSpacing(25)

        hbox = QHBoxLayout()
        checkbox = WidgetFactory.get_widget("checkbox", window=self)
        label = QLabel("Translation", self)
        hbox.addWidget(checkbox)
        hbox.addWidget(label)
        self.t_layout.addLayout(hbox)
        self.checkbox_label_pairs.append((checkbox, label))
        for axis in ["x", "y", "z"]:
            sub_layout = QHBoxLayout()
            label = QLabel("t_" + axis)
            label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            sub_layout.addWidget(label)
            spin_box = WidgetFactory.get_widget(
                "double_spin_box",
                window=self,
                name="t_spin_box_" + axis,
                step=self.GUI_config.t_layout_step,
                min=self.GUI_config.t_layout_min,
                max=self.GUI_config.t_layout_max,
                value=self.GUI_config.t_layout_default[axis],
                decimals=self.GUI_config.t_layout_decimal,
            )
            sub_layout.addWidget(spin_box)
            self.t_layout.addLayout(sub_layout)
            self.t_l2c_spin_boxes.append(spin_box)

        # Translation learning rate
        lr_layout = QHBoxLayout()
        label = QLabel("lr")
        label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        lr_layout.addWidget(label)
        self.t_lr_input = WidgetFactory.get_widget(
            "lr_text_input", value=str(self.training_config.t_vec_lr)
        )
        lr_layout.addWidget(self.t_lr_input)
        self.t_layout.addLayout(lr_layout)
        pose_params_layout.addLayout(self.t_layout)
        pose_layout.addLayout(pose_params_layout)
        pose_frame.setLayout(pose_layout)
        self.param_layout.addWidget(pose_frame)
        self.param_layout.addSpacing(10)

        # Light parameters frame
        light_frame = QFrame()
        light_frame.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        light_layout = QVBoxLayout()
        label = QLabel("Light Parameters")
        label.setFont(font)
        light_layout.addWidget(label, alignment=Qt.AlignTop)

        # Choose light type
        dropdownlist_layout = QVBoxLayout()
        self.light_comboBox = WidgetFactory.get_widget(
            "combo_box", 
            window=self,
            items=["Point Light Source", "Gaussian 1D", "Gaussian 2D", "1D MLP", "2D MLP"],
            current_text=self.GUI_config.light_source_default,
            onchange=self.check_light_comboBox
        )
        dropdownlist_layout.addWidget(self.light_comboBox)

        # Light parameters
        sub_layout = QHBoxLayout()
        self.sub_layout_left = QHBoxLayout()
        self.sub_layout_right = QVBoxLayout()

        checkbox = WidgetFactory.get_widget("checkbox", window=self)
        label = QLabel("\u03C3_x")
        self.sub_layout_left.addWidget(checkbox)
        self.checkbox_label_pairs.append((checkbox, label))
        sub_layout.addLayout(self.sub_layout_left)

        sub_layout_sigma_x = QHBoxLayout()
        self.sigma_x_label = QLabel("\u03C3_x")
        sub_layout_sigma_x.addWidget(self.sigma_x_label)
        self.sigma_x_spin_box = WidgetFactory.get_widget(
            "double_spin_box",
            window=self,
            name="sigma_x_spin_box",
            step=self.GUI_config.sigma_step,
            min=self.GUI_config.sigma_min,
            max=self.GUI_config.sigma_max,
            value=self.GUI_config.sigma_default,
            decimals=self.GUI_config.sigma_decimal,
        )
        sub_layout_sigma_x.addWidget(self.sigma_x_spin_box, alignment=Qt.AlignRight)
        sub_layout_sigma_x.addSpacing(25)
        self.sub_layout_right.addLayout(sub_layout_sigma_x)

        sub_layout_sigma_y = QHBoxLayout()
        label = QLabel("\u03C3_y")
        sub_layout_sigma_y.addWidget(label)
        self.sigma_y_spin_box = WidgetFactory.get_widget(
            "double_spin_box",
            window=self,
            name="sigma_y_spin_box",
            step=self.GUI_config.sigma_step,
            min=self.GUI_config.sigma_min,
            max=self.GUI_config.sigma_max,
            value=self.GUI_config.sigma_default,
            decimals=self.GUI_config.sigma_decimal,
        )
        sub_layout_sigma_y.addWidget(self.sigma_y_spin_box, alignment=Qt.AlignRight)
        sub_layout_sigma_y.addSpacing(25)
        self.sub_layout_right.addLayout(sub_layout_sigma_y)

        sub_layout.addLayout(self.sub_layout_right)
        dropdownlist_layout.addLayout(sub_layout)

        self.mlp_hbox = QHBoxLayout()
        checkbox = WidgetFactory.get_widget("checkbox", window=self)
        label = QLabel("MLP parameters", self)
        self.mlp_hbox.addWidget(checkbox)
        self.mlp_hbox.addWidget(label)
        self.checkbox_label_pairs.append((checkbox, label))
        self.mlp_hbox.addSpacing(25)
        dropdownlist_layout.addLayout(self.mlp_hbox)

        # Light learning rate
        lr_layout = QHBoxLayout()
        label = QLabel("lr")
        label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        lr_layout.addWidget(label)
        self.light_lr_input = WidgetFactory.get_widget(
            "lr_text_input", value=str(self.training_config.lightmodel_lr)
        )
        lr_layout.addWidget(self.light_lr_input)
        dropdownlist_layout.addLayout(lr_layout)
        light_layout.addLayout(dropdownlist_layout)
        light_frame.setLayout(light_layout)
        self.param_layout.addWidget(light_frame)
        self.param_layout.addSpacing(10)

        # Image settings frame
        image_frame = QFrame()
        image_frame.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        image_layout = QVBoxLayout()
        label = QLabel("Image Settings")
        label.setFont(font)
        image_layout.addWidget(label, alignment=Qt.AlignTop)
        image_layout.addSpacing(10)

        img_column_layout = QVBoxLayout()

        # Dataset
        self.data_layout = QVBoxLayout()
        label = QLabel("Dataset")
        self.data_layout.addWidget(label)
        self.data_comboBox = WidgetFactory.get_widget(
            "combo_box", 
            window=self,
            items=["Training set", "Testing set"],
            current_text=self.GUI_config.light_source_default,
            onchange=self.check_data_comboBox
        )
        self.data_layout.addWidget(self.data_comboBox)
        self.data_layout.addStretch()
        img_column_layout.addLayout(self.data_layout)

        # Pagination
        self.page_layout = QVBoxLayout()
        label = QLabel("Image " + str(self.GUI_config.page_default) + " / " + str(len(self.current_dataset)))
        spin_box = QSpinBox()
        self.page_layout.addWidget(label)
        spin_box = WidgetFactory.get_widget(
            "spin_box",
            window=self,
            name="page_spin_box",
            step=self.GUI_config.page_step,
            min=self.GUI_config.page_min,
            max=self.GUI_config.page_max,
            value=self.GUI_config.page_default,
        )
        spin_box.valueChanged.connect(self.update_page_label)
        self.page_layout.addWidget(spin_box)
        self.page_layout.addStretch()
        img_column_layout.addLayout(self.page_layout)

        # Zoom
        zoom_layout = QHBoxLayout()
        label = QLabel("zoom")
        zoom_layout.addWidget(label)
        self.zoom_spin_box = WidgetFactory.get_widget(
            "double_spin_box",
            window=self,
            name="zoom_spin_box",
            step=self.GUI_config.zoom_step,
            min=self.GUI_config.zoom_min,
            max=self.GUI_config.zoom_max,
            value=self.GUI_config.zoom_default,
        )
        zoom_layout.addWidget(self.zoom_spin_box)
        zoom_layout.addStretch()
        img_column_layout.addLayout(zoom_layout)
        image_layout.addLayout(img_column_layout)
        image_frame.setLayout(image_layout)
        self.param_layout.addWidget(image_frame)
        self.param_layout.addSpacing(10)

        # Training frame
        training_frame = QFrame()
        training_frame.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        training_layout = QVBoxLayout()
        label = QLabel("Training Info")
        label.setFont(font)
        training_layout.addWidget(label, alignment=Qt.AlignTop)

        training_column_layout = QVBoxLayout()
 
        # Error
        error_layout = QVBoxLayout()
        self.error_label = QLabel("error")
        error_layout.addWidget(self.error_label)
        error_layout.addSpacing(10)
        training_column_layout.addLayout(error_layout)

        # Progress bar
        progress_bar_layout = QVBoxLayout()
        label = QLabel("Training Progress")
        progress_bar_layout.addWidget(label)
        self.progressBar = WidgetFactory.get_widget(
            "progress_bar", 
            window=self, 
            max=self.training_config.epoch
        )
        progress_bar_layout.addWidget(self.progressBar)
        training_column_layout.addLayout(progress_bar_layout)

        # Num of Epochs
        self.num_epoch_layout = QVBoxLayout()
        label = QLabel("Num of Epochs")
        self.num_epoch_layout.addWidget(label)
        spin_box = WidgetFactory.get_widget(
            "spin_box",
            window=self,
            name="num_epoch_spin_box",
            step=self.GUI_config.num_epoch_step,
            min=self.GUI_config.training_update_times,
            max=self.GUI_config.num_epoch_max,
            value=self.training_config.epoch,
        )
        self.num_epoch_layout.addWidget(spin_box)
        training_column_layout.addLayout(self.num_epoch_layout)
        training_layout.addLayout(training_column_layout)
        training_frame.setLayout(training_layout)
        self.param_layout.addWidget(training_frame)
        self.param_layout.addSpacing(25)
        self.param_layout.addStretch()

        cali_layout = QVBoxLayout()

        # Status
        status_layout = QVBoxLayout()
        self.status_label = QLabel("")
        status_layout.addWidget(self.status_label)
        status_layout.addStretch()
        cali_layout.addLayout(status_layout)

        # Save and load model parameters
        self.save_param_button = WidgetFactory.get_widget(
            "button",
            name="SAVE",
            base_color="#6690cc",
            onclick=self.save_model_param,
        )
        save_param_button_layout = QVBoxLayout()
        save_param_button_layout.addStretch()
        save_param_button_layout.addWidget(self.save_param_button)
        cali_layout.addLayout(save_param_button_layout)

        self.load_param_button = WidgetFactory.get_widget(
            "button",
            name="LOAD",
            base_color="#6690cc",
            onclick=self.load_model_param,
        )
        load_param_button_layout = QVBoxLayout()
        load_param_button_layout.addStretch()
        load_param_button_layout.addWidget(self.load_param_button)
        cali_layout.addLayout(load_param_button_layout)

        # Update rendered image button
        self.update_button = WidgetFactory.get_widget(
            "button",
            name="UPDATE",
            base_color="#66cc75",
            onclick=self.onclick_update,
        )
        update_button_layout = QVBoxLayout()
        update_button_layout.addStretch()
        update_button_layout.addWidget(self.update_button)
        cali_layout.addLayout(update_button_layout)

        # Start training button
        self.start_button = WidgetFactory.get_widget(
            "button",
            name="START",
            base_color="#66cc75",
            onclick=self.onclick_start,
        )
        start_button_layout = QVBoxLayout()
        start_button_layout.addStretch()
        start_button_layout.addWidget(self.start_button)
        cali_layout.addLayout(start_button_layout)

        # Stop training button
        self.stop_button = WidgetFactory.get_widget(
            "button",
            name="STOP",
            base_color="#cc6f66",
            onclick=self.onclick_stop,
        )
        stop_button_layout = QVBoxLayout()
        stop_button_layout.addStretch()
        stop_button_layout.addWidget(self.stop_button)
        cali_layout.addLayout(stop_button_layout)

        # Save rendered image button
        self.save_img_button = WidgetFactory.get_widget(
            "button",
            name="SAVE IMG",
            base_color="#b466cc",
            onclick=self.onclick_save_img,
        )
        save_img_button_layout = QVBoxLayout()
        save_img_button_layout.addStretch()
        save_img_button_layout.addWidget(self.save_img_button)
        cali_layout.addLayout(save_img_button_layout)

        self.param_layout.addLayout(cali_layout)

        # Add the whole parameters setup panel
        layout.addLayout(self.param_layout)
        self.setWindowTitle("CAMERA-LIGHT CALIBRATION")

    def save_model_param(self)->None:
        print("Saving model parameters...")
        save_dict = {
            'model_state_dict': self.shading_model.state_dict(),
            'so3': self.shading_model.light._r_l2c_SO3.log()}
        res = torch.save(save_dict, 'model_parameters.pth')
        print("Parameters saved!")

    def load_model_param(self)->None:
        print("Loading model parameters...")
        self.onclick_update()
        dict = torch.load('model_parameters.pth')
        self.shading_model.load_state_dict(dict['model_state_dict'])
        r_vec = dict['so3'].squeeze()
        self.shading_model.light.set_r_vec(tuple([r_vec[0], r_vec[1], r_vec[2]]))
        if hasattr(self.shading_model.light, 'sigma'):
            if self.shading_model.light.sigma.ndim == 0:
                self.update_shading_model_param(self.shading_model.albedo, self.shading_model.light.gamma, self.shading_model.light.tau, self.shading_model.ambient_light, self.shading_model.light._t_vec, self.shading_model.light._r_l2c_SO3.log(), [self.shading_model.light.sigma, 0])
            else:
                self.update_shading_model_param(self.shading_model.albedo, self.shading_model.light.gamma, self.shading_model.light.tau, self.shading_model.ambient_light, self.shading_model.light._t_vec, self.shading_model.light._r_l2c_SO3.log(), [self.shading_model.light.sigma[0], self.shading_model.light.sigma[1]])
        else:
            self.update_shading_model_param(self.shading_model.albedo, self.shading_model.light.gamma, self.shading_model.light.tau, self.shading_model.ambient_light, self.shading_model.light._t_vec, self.shading_model.light._r_l2c_SO3.log(), [0, 0])
        print("Parameters loaded!")
        self.onclick_update()

    def onclick_save_img(self):
        imgs_raw, imgs_rendered = self.render()
        if not os.path.exists("result"):
            os.makedirs("result")
        for i, img in enumerate(imgs_raw):
            cv.imwrite(f'result/{TrainingConfig.image_path.split("/")[-1]}_raw_img_{i}.png', img)
        for i, img in enumerate(imgs_rendered):
            cv.imwrite(f'result/{TrainingConfig.image_path.split("/")[-1]}_rendered_img_{i}.png', img)
        print("Images saved!")

    def check_selected_parameters(self):
        checked_parameters = []
        for checkbox, label in self.checkbox_label_pairs:
            if checkbox.isChecked() and checkbox.isVisible():
                checked_parameters.append(label.text())
        return checked_parameters

    def show_sigma_x(self):
        self.sub_layout_left.itemAt(0).widget().show()
        self.sub_layout_right.itemAt(0).itemAt(0).widget().show()
        self.sub_layout_right.itemAt(0).itemAt(1).widget().show()

    def hide_sigma_x(self):
        self.sub_layout_left.itemAt(0).widget().hide()
        self.sub_layout_right.itemAt(0).itemAt(0).widget().hide()
        self.sub_layout_right.itemAt(0).itemAt(1).widget().hide()

    def show_sigma_y(self):
        self.sub_layout_right.itemAt(1).itemAt(0).widget().show()
        self.sub_layout_right.itemAt(1).itemAt(1).widget().show()

    def hide_sigma_y(self):
        self.sub_layout_right.itemAt(1).itemAt(0).widget().hide()
        self.sub_layout_right.itemAt(1).itemAt(1).widget().hide()

    def show_MLP_parameters(self):
        self.mlp_hbox.itemAt(0).widget().show()
        self.mlp_hbox.itemAt(1).widget().show()

    def hide_MLP_parameters(self):
        self.mlp_hbox.itemAt(0).widget().hide()
        self.mlp_hbox.itemAt(1).widget().hide()

    def check_light_comboBox(self):
        if self.get_light_source() == "Point Light Source":
            self.hide_sigma_x()
            self.hide_sigma_y()
            self.hide_MLP_parameters()
        elif self.get_light_source() == "1D MLP" or self.get_light_source() == "2D MLP":
            self.hide_sigma_x()
            self.hide_sigma_y()
            self.show_MLP_parameters()
        elif self.get_light_source() == "Gaussian 1D":
            self.sigma_x_label.setText("\u03C3")
            self.show_sigma_x()
            self.hide_sigma_y()
            self.hide_MLP_parameters()
        elif self.get_light_source() == "Gaussian 2D":
            self.sigma_x_label.setText("\u03C3_x")
            self.show_sigma_x()
            self.show_sigma_y()
            self.hide_MLP_parameters()

    def check_data_comboBox(self):
        if self.get_cur_dataset() == "Training set":
            self.current_dataset = self.training_dataset
            self.current_dataloader = self.training_dataloader
        elif self.get_cur_dataset() == "Testing set":
            self.current_dataset = self.testing_dataset
            self.current_dataloader = self.testing_dataloader
        self.update_page_label(self.GUI_config.page_default)
        self.onclick_update()

    def set_shading_model(self):
        self.shading_model = ShadingModel(
            brdf=self.training_config.brdf,
            light=self.get_light_source().replace(" ", ""),
            device=self.training_config.device,
        )
        self.shading_model.to(device=self.training_config.device)

    def update_page_label(self, value):
        self.page_layout.itemAt(1).widget().setMaximum(len(self.current_dataset))
        self.page_layout.itemAt(0).widget().setText("Image " + str(value) + " / " + str(len(self.current_dataset)))

    def update_sigma_label(self, value):
        if self.get_light_source() == "Gaussian 1D":
            self.sigma_x_label.setText("\u03C3")
        elif self.get_light_source() == "Gaussian 2D":
            self.sigma_x_label.setText("\u03C3_x")

    def get_page_number(self):
        return self.page_layout.itemAt(1).widget().value()

    def get_zoom_ratio(self):
        return self.zoom_spin_box.value()

    def get_num_epoch(self):
        num_epoch = self.num_epoch_layout.itemAt(1).widget().value()
        self.progressBar.setMaximum(num_epoch)
        return num_epoch

    def set_img(self, im: np.ndarray, flag: str = "raw") -> bool:
        img = cv.pyrDown(im)
        height, width = img.shape
        height, width = img.shape
        zoom_ratio = self.get_zoom_ratio()
        width = int(width * zoom_ratio)
        height = int(height * zoom_ratio)
        img = cv.resize(img, (width, height))
        bytesPerLine = width
        qImg = QImage(img.data, width, height, bytesPerLine, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qImg)
        if flag == "raw":
            self.image_label_raw.setPixmap(pixmap)
        elif flag == "rendered":
            self.image_label_rendered.setPixmap(pixmap)
        elif flag == "diff":
            self.image_label_diff.setPixmap(pixmap)
        else:
            raise NotImplementedError("Only setting raw image or rendered image")

    def get_albedo(self) -> float:
        return self.albedo_spin_box.value()

    def get_gamma(self) -> float:
        return self.gamma_spin_box.value()
    
    def get_tau(self) -> float:
        return self.tau_spin_box.value()

    def get_translation_vector(self) -> tuple[float, float, float]:
        return tuple([self.t_layout.itemAt(i+1).itemAt(1).widget().value() for i in range(3)])
    
    def get_rotation_vector(self) -> tuple[float, float, float]:
        return tuple([self.r_layout.itemAt(i+1).itemAt(1).widget().value() for i in range(3)])
    
    def get_sigma(self) -> list[float]:
        return [self.sigma_x_spin_box.value(), self.sigma_y_spin_box.value()]
    
    def get_ambient(self) -> float:
        return self.ambient_spin_box.value()

    def get_light_source(self) -> string:
        return self.light_comboBox.currentText()
    
    def get_cur_dataset(self) -> string:
        return self.data_comboBox.currentText()

    def onclick_stop(self):
        if self.training_thread:
            self.training_thread.stop()

    def on_training_stopped(self):
        print("Training stopped")

    def onclick_update(self):
        self.get_num_epoch()
        self.check_selected_parameters()
        if self.shading_model.light.name != self.get_light_source():
            self.set_shading_model()
        self.update_shading_model(self.get_albedo(), self.get_gamma(), self.get_tau(), self.get_ambient(), self.get_translation_vector(), self.get_rotation_vector(), self.get_sigma())
        imgs_raw, imgs_rendered = self.render()
        self.update_images(imgs_raw, imgs_rendered)
        QApplication.processEvents()
        self.resize(self.minimumSizeHint())
        self.adjustSize()

    def update_images(self, imgs_raw, imgs_rendered):
        image_index = self.get_page_number() - 1
        self.set_img(imgs_raw[image_index], flag="raw")
        self.set_img(imgs_rendered[image_index], flag="rendered")
        diff = np.clip(np.round(((imgs_rendered[image_index].astype(np.int16) - imgs_raw[image_index].astype(np.int16)) + 255) * 0.5).astype(np.uint8), 0, 255)
        self.update_error(imgs_raw, imgs_rendered)
        self.set_img(diff, flag="diff")

    def enable_start_button(self):
        self.update_status("Training\nstopped")
        self.start_button.setEnabled(True)
        set_button_style(self.start_button, "#66cc75")
        self.onclick_update()

    def disable_start_button(self):
        self.update_status("Training\nin progress")
        self.start_button.setEnabled(False)
        set_button_style(self.start_button, "#999999")

    def enable_update_button(self):
        self.update_button.setEnabled(True)
        set_button_style(self.update_button, "#66cc75")

    def disable_update_button(self):
        self.update_button.setEnabled(False)
        set_button_style(self.update_button, "#999999")

    def get_all_lr(self):
        return {
            "Albedo": float(self.albedo_lr_input.text()),
            "gamma": float(self.gamma_lr_input.text()),
            "tau": float(self.tau_lr_input.text()),
            "Ambient": float(self.ambient_lr_input.text()),
            "Rotation": float(self.r_lr_input.text()),
            "Translation": float(self.t_lr_input.text()),
            "\u03C3_x": float(self.light_lr_input.text()),
            "MLP parameters": float(self.light_lr_input.text())
        }

    def onclick_start(self):
        self.onclick_update()
        self.update_progress_bar(0)
        self.disable_start_button()
        self.disable_update_button()
        self.training_thread = TrainingThread(self.shading_model, self.training_dataloader, self.render, self.check_selected_parameters(), self.get_num_epoch(), self.get_all_lr())
        self.training_thread.update_images.connect(self.update_images)
        self.training_thread.update_shading_model_param.connect(self.update_shading_model_param)
        self.training_thread.update_progress_bar.connect(self.update_progress_bar)
        self.training_thread.training_stopped.connect(self.enable_start_button)
        self.training_thread.training_stopped.connect(self.enable_update_button)
        self.training_thread.start()
        
    def update_shading_model(self, albedo_value: float, gamma_value: float, tau_value: float, ambient_value: float, t_vec: tuple[float, float, float], r_vec: tuple[float, float, float], sigma: list[float]):
        self.shading_model.set_albedo(albedo_value)
        self.shading_model.light.set_gamma(gamma_value)
        self.shading_model.light.set_tau(tau_value)
        self.shading_model.set_ambient_light(ambient_value)
        self.shading_model.light.set_t_vec(t_vec)
        self.shading_model.light.set_r_vec(r_vec)
        self.shading_model.light.set_sigma(sigma) 

    def update_shading_model_param(self, albedo_value: float, gamma_value: float, tau_value: float, ambient_value: float, t_vec: torch.Tensor, r_vec: torch.Tensor, sigma: list[float]):
        self.albedo_spin_box.setValue(albedo_value)
        self.gamma_spin_box.setValue(gamma_value)
        self.tau_spin_box.setValue(tau_value)
        self.ambient_spin_box.setValue(ambient_value)
        for i in range(3):
            self.r_layout.itemAt(i+1).itemAt(1).widget().setValue(r_vec[0][0][i])
            self.t_layout.itemAt(i+1).itemAt(1).widget().setValue(t_vec[0][0][i])
        self.sigma_x_spin_box.setValue(sigma[0])
        self.sigma_y_spin_box.setValue(sigma[1])

    def update_error(self, imgs_raw: np.ndarray, imgs_rendered: np.ndarray):
        error = 0.
        for i in range(len(imgs_raw)):
            diff = (imgs_rendered[i].astype(np.int16) - imgs_raw[i].astype(np.int16)).astype(np.uint8)
            error += np.square(diff).mean()
        self.error_label.setText("error: " + str(round(error / len(imgs_raw), self.GUI_config.error_decimal)))

    def update_progress_bar(self, value):
        self.progressBar.setValue(value)

    def update_status(self, status: str):
        self.status_label.setText(status)

    def render(self):
        imgs_raw = []
        imgs_rendered = []
        for itr, (pts, intensities, rvec_w2c, tvec_w2c, img, mask) in enumerate(
            self.current_dataloader
        ):
            rendered_intensities = self.shading_model(pts, rvec_w2c, tvec_w2c)
            img_raw = cv.pyrDown(img.squeeze().cpu().numpy().astype(np.uint8))
            img.view(-1)[mask.view(-1)] = rendered_intensities.to(torch.uint8)
            img_viz = img.squeeze().cpu().numpy().astype(np.uint8)
            img_viz = cv.pyrDown(img_viz)

            imgs_raw.append(img_raw)
            imgs_rendered.append(img_viz)

        return imgs_raw, imgs_rendered


def main():
    app = QApplication(sys.argv)
    main_window = MainWindow()
    min_height = main_window.height()
    min_width = main_window.width() 
    screen = QApplication.primaryScreen()
    screen_geometry = screen.geometry()
    screen_width = screen_geometry.width()
    main_window.setGeometry(100, 100, min(screen_width, min_width), min(300, min_height))

    gui_config = GUIConfig()

    default_t_vec = (gui_config.t_layout_default["x"], gui_config.t_layout_default["y"], gui_config.t_layout_default["z"])
    default_r_vec = (gui_config.r_layout_default["x"], gui_config.r_layout_default["y"], gui_config.r_layout_default["z"])
    main_window.update_shading_model(gui_config.albedo_default, gui_config.gamma_default, gui_config.tau_default, gui_config.ambient_default, default_t_vec, default_r_vec, [gui_config.sigma_default, gui_config.sigma_default])
    
    imgs_raw, imgs_rendered = main_window.render()
    main_window.page_layout.itemAt(1).widget().setMaximum(len(imgs_raw))
    main_window.set_img(imgs_raw[0], flag="raw")
    main_window.set_img(imgs_rendered[0], flag="rendered")
    diff = np.clip(np.round(((imgs_rendered[0].astype(np.int16) - imgs_raw[0].astype(np.int16)) + 255) * 0.5).astype(np.uint8), 0, 255)
    main_window.update_error(imgs_raw, imgs_rendered)
    main_window.set_img(diff, flag="diff")
    main_window.check_light_comboBox()
    main_window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
