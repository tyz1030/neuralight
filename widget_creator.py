from PySide6.QtWidgets import (
    QPushButton,
    QLineEdit,
    QSpinBox,
    QDoubleSpinBox,
    QLabel,
    QSizePolicy,
    QCheckBox,
    QProgressBar,
    QComboBox
)
from PySide6.QtGui import QDoubleValidator, QColor, QPixmap
from PySide6.QtCore import Qt

from config import GUIConfig


class WidgetCreator:
    """Abstract creator class."""

    def create_widget(self, **kwargs):
        raise NotImplementedError("Factory method is not implemented.")


class ButtonCreator(WidgetCreator):
    """Concrete creator for a button."""
    
    def create_widget(self, base_color='#66ccff', **kwargs):
        button = QPushButton(kwargs.get("name", "Button"))
        button = set_button_style(button, base_color)
        button.clicked.connect(kwargs.get("onclick", lambda: print("Button clicked.")))
        return button


class LRInputCreator(WidgetCreator):
    """Concrete creator for a text input."""

    def create_widget(self, **kwargs):
        line_edit = QLineEdit(kwargs.get("value", "0.01"))   
        double_validator = QDoubleValidator()
        line_edit.setValidator(double_validator)
        line_edit.setFixedWidth(GUIConfig.lr_input_width)
        return line_edit


class SpinBoxCreator(WidgetCreator):
    """Concrete creator for a spin box."""

    def create_widget(self, **kwargs):
        spin_box = QSpinBox(kwargs.get("window"))
        spin_box.setObjectName(kwargs.get("name", "spin_box"))
        spin_box.setSingleStep(kwargs.get("step", 1))
        spin_box.setMinimum(kwargs.get("min", 0))
        spin_box.setMaximum(kwargs.get("max", 100))
        spin_box.setValue(kwargs.get("value", 0))
        return spin_box


class DoubleSpinBoxCreator(WidgetCreator):
    """Concrete creator for a double spin box."""

    def create_widget(self, **kwargs):
        spin_box = QDoubleSpinBox(kwargs.get("window"))
        spin_box.setObjectName(kwargs.get("name", "spin_box"))
        spin_box.setSingleStep(kwargs.get("step", 0.1))
        spin_box.setMinimum(kwargs.get("min", 0))
        spin_box.setMaximum(kwargs.get("max", 100.))
        spin_box.setValue(kwargs.get("value", 0))
        spin_box.setDecimals(kwargs.get("decimals", 2))
        spin_box.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        return spin_box


class ImageLabelCreator(WidgetCreator):
    """Concrete creator for an image label."""

    def create_widget(self, **kwargs):
        image_label = QLabel(kwargs.get("window"))
        image_label.setPixmap(QPixmap(kwargs.get("path", "path_to_your_image.jpg")))
        image_label.setAlignment(Qt.AlignCenter)
        return image_label
    

class CheckboxCreator(WidgetCreator):
    """Concrete creator for a checkbox."""

    def create_widget(self, **kwargs):
        checkbox = QCheckBox(kwargs.get("window"))
        checkbox.setChecked(kwargs.get("checked", True))
        return checkbox
    

class ProgressBarCreator(WidgetCreator):
    """Concrete creator for a progress bar."""

    def create_widget(self, **kwargs):
        progressBar = QProgressBar(kwargs.get("window"))
        progressBar.setMaximum(kwargs.get("max", 1000))
        progressBar.setFixedWidth(GUIConfig.progress_bar_length)
        return progressBar
    

class ComboBoxCreator(WidgetCreator):
    """Concrete creator for a combo box."""

    def create_widget(self, **kwargs):
        combo_box = QComboBox(kwargs.get("window"))
        combo_box.addItems(kwargs.get("items", []))
        combo_box.setCurrentText(kwargs.get("current_text", ""))
        combo_box.currentIndexChanged.connect(kwargs.get("onchange", lambda: print("Combo box changed.")))
        return combo_box


class WidgetFactory:
    """Factory class to get widgets."""

    @staticmethod
    def get_widget(widget_type, **kwargs):
        creators = {
            "button": ButtonCreator(),
            "lr_text_input": LRInputCreator(),
            "double_spin_box": DoubleSpinBoxCreator(),
            "image_label": ImageLabelCreator(),
            "spin_box": SpinBoxCreator(),
            "checkbox": CheckboxCreator(),
            "progress_bar": ProgressBarCreator(),
            "combo_box": ComboBoxCreator()
        }
        return creators.get(widget_type).create_widget(**kwargs)

def adjust_color_brightness(color, factor):
        """ Adjust the color brightness. Factor > 1 will lighten; factor < 1 will darken. """
        color = QColor(color)
        red = color.red() * factor
        green = color.green() * factor
        blue = color.blue() * factor

        red = min(255, max(0, red))
        green = min(255, max(0, green))
        blue = min(255, max(0, blue))

        return f'rgb({int(red)}, {int(green)}, {int(blue)})'

def set_button_style(button, base_color):
        hover_color = adjust_color_brightness(base_color, 1.1)
        pressed_color = adjust_color_brightness(base_color, 0.9)
        button.setStyleSheet(f"""
            QPushButton {{
                color: white;
                background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                                stop:0 {base_color}, stop:1 {base_color});
                border-radius: 5px;
                padding: 6px 10px;
                font-size: 12px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {hover_color};
            }}
            QPushButton:pressed {{
                background-color: {pressed_color};
            }}
        """)
        return button
