"""Qt stylesheet and color palette for the interactive UI."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ColorPalette:
    """Color palette for the application theme."""

    # Base colors
    bg_primary: str  # Main background
    bg_secondary: str  # Secondary background (panels)
    bg_tertiary: str  # Tertiary background (inputs)
    bg_hover: str  # Hover state background

    # Text colors
    text_primary: str  # Primary text
    text_secondary: str  # Secondary text (labels)
    text_hint: str  # Hint/placeholder text
    text_disabled: str  # Disabled text

    # Border colors
    border_normal: str  # Normal borders
    border_focus: str  # Focused element border
    border_hover: str  # Hovered element border

    # Accent colors
    accent_primary: str  # Primary accent (buttons, focus)
    accent_hover: str  # Accent hover state
    accent_pressed: str  # Accent pressed state

    # Semantic colors
    success: str
    warning: str
    error: str
    info: str


# Dark theme palette (refined from current contrast theme)
DARK_PALETTE = ColorPalette(
    # Backgrounds - slightly lighter than current for better hierarchy
    bg_primary="#1a1a1a",  # Main window background
    bg_secondary="#242424",  # Panel backgrounds
    bg_tertiary="#2d2d2d",  # Input backgrounds
    bg_hover="#333333",  # Hover states
    # Text - improved contrast
    text_primary="#e8e8e8",  # Main text
    text_secondary="#b8b8b8",  # Labels
    text_hint="#888888",  # Hints/placeholders
    text_disabled="#555555",  # Disabled
    # Borders - more visible
    border_normal="#404040",  # Normal borders
    border_focus="#ff7600",  # Focus (orange accent)
    border_hover="#505050",  # Hover
    # Accent - orange theme from matplotlib
    accent_primary="#ff7600",
    accent_hover="#ff8820",
    accent_pressed="#e06800",
    # Semantic
    success="#4caf50",
    warning="#ff9800",
    error="#f44336",
    info="#2196f3",
)


# Light theme palette (refined from default theme)
LIGHT_PALETTE = ColorPalette(
    # Backgrounds
    bg_primary="#f5f5f5",
    bg_secondary="#ffffff",
    bg_tertiary="#fafafa",
    bg_hover="#eeeeee",
    # Text
    text_primary="#212121",
    text_secondary="#666666",
    text_hint="#999999",
    text_disabled="#cccccc",
    # Borders
    border_normal="#d0d0d0",
    border_focus="#1f77b4",
    border_hover="#b0b0b0",
    # Accent - blue theme
    accent_primary="#1f77b4",
    accent_hover="#3489c6",
    accent_pressed="#1a6ba0",
    # Semantic
    success="#4caf50",
    warning="#ff9800",
    error="#f44336",
    info="#2196f3",
)


def generate_stylesheet(palette: ColorPalette) -> str:
    """Generate a comprehensive Qt stylesheet using the given palette."""
    return f"""
    /* ===== Global Defaults ===== */
    QWidget {{
        background-color: {palette.bg_primary};
        color: {palette.text_primary};
        font-size: 13px;
    }}

    /* ===== Group Boxes (Panel Groups) ===== */
    QGroupBox {{
        background-color: {palette.bg_secondary};
        border: 1px solid {palette.border_normal};
        border-radius: 10px;
        margin-top: 24px;
        padding: 16px;
        font-weight: 600;
        font-size: 13px;
    }}

    QGroupBox::title {{
        subcontrol-origin: margin;
        subcontrol-position: top left;
        left: 16px;
        top: 8px;
        padding: 6px 12px;
        background-color: {palette.bg_secondary};
        color: {palette.text_primary};
    }}

    /* ===== Labels ===== */
    QLabel {{
        background-color: transparent;
        color: {palette.text_secondary};
        padding: 2px;
    }}

    QLabel[class="hint"] {{
        color: {palette.text_hint};
        font-style: italic;
        font-size: 12px;
    }}

    QLabel[class="error"] {{
        color: {palette.error};
        font-weight: 500;
    }}

    QLabel[class="success"] {{
        color: {palette.success};
        font-weight: 500;
    }}

    /* ===== Push Buttons ===== */
    QPushButton {{
        background-color: {palette.bg_tertiary};
        border: 1px solid {palette.border_normal};
        border-radius: 6px;
        padding: 8px 16px;
        color: {palette.text_primary};
        font-weight: 500;
    }}

    QPushButton:hover {{
        background-color: {palette.bg_hover};
        border-color: {palette.border_hover};
    }}

    QPushButton:pressed {{
        background-color: {palette.border_normal};
    }}

    QPushButton:disabled {{
        background-color: {palette.bg_secondary};
        color: {palette.text_disabled};
        border-color: {palette.border_normal};
    }}

    QPushButton[class="primary"] {{
        background-color: {palette.accent_primary};
        border-color: {palette.accent_primary};
        color: white;
        font-weight: 600;
    }}

    QPushButton[class="primary"]:hover {{
        background-color: {palette.accent_hover};
        border-color: {palette.accent_hover};
    }}

    QPushButton[class="primary"]:pressed {{
        background-color: {palette.accent_pressed};
        border-color: {palette.accent_pressed};
    }}

    QPushButton[class="danger"] {{
        border-color: {palette.error};
        color: {palette.error};
    }}

    QPushButton[class="danger"]:hover {{
        background-color: {palette.error};
        color: white;
    }}

    /* ===== Line Edits (Text Inputs) ===== */
    QLineEdit {{
        background-color: {palette.bg_tertiary};
        border: 1px solid {palette.border_normal};
        border-radius: 5px;
        padding: 7px 10px;
        color: {palette.text_primary};
        selection-background-color: {palette.accent_primary};
    }}

    QLineEdit:hover {{
        border-color: {palette.border_hover};
    }}

    QLineEdit:focus {{
        border: 2px solid {palette.border_focus};
        padding: 6px 9px;
    }}

    QLineEdit:disabled {{
        background-color: {palette.bg_secondary};
        color: {palette.text_disabled};
    }}

    QLineEdit::placeholder {{
        color: {palette.text_hint};
        font-style: italic;
    }}

    /* ===== Combo Boxes (Dropdowns) ===== */
    QComboBox {{
        background-color: {palette.bg_tertiary};
        border: 1px solid {palette.border_normal};
        border-radius: 5px;
        padding: 7px 10px;
        color: {palette.text_primary};
    }}

    QComboBox:hover {{
        border-color: {palette.border_hover};
    }}

    QComboBox:focus {{
        border: 2px solid {palette.border_focus};
        padding: 6px 9px;
    }}

    QComboBox:disabled {{
        background-color: {palette.bg_secondary};
        color: {palette.text_disabled};
    }}

    QComboBox::drop-down {{
        border: none;
        width: 30px;
    }}

    QComboBox::down-arrow {{
        image: none;
        border-left: 4px solid transparent;
        border-right: 4px solid transparent;
        border-top: 6px solid {palette.text_secondary};
        margin-right: 8px;
    }}

    QComboBox::down-arrow:hover {{
        border-top-color: {palette.text_primary};
    }}

    QComboBox QAbstractItemView {{
        background-color: {palette.bg_tertiary};
        border: 1px solid {palette.border_normal};
        border-radius: 5px;
        selection-background-color: {palette.accent_primary};
        selection-color: white;
        padding: 4px;
    }}

    QComboBox QAbstractItemView::item {{
        padding: 6px 10px;
        border-radius: 3px;
    }}

    QComboBox QAbstractItemView::item:hover {{
        background-color: {palette.bg_hover};
    }}

    /* ===== Spin Boxes ===== */
    QSpinBox, QDoubleSpinBox {{
        background-color: {palette.bg_tertiary};
        border: 1px solid {palette.border_normal};
        border-radius: 5px;
        padding: 7px 10px;
        padding-right: 28px;  /* Make room for wider buttons */
        color: {palette.text_primary};
        min-width: 80px;  /* Ensure enough space for numbers */
    }}

    QSpinBox:hover, QDoubleSpinBox:hover {{
        border-color: {palette.border_hover};
    }}

    QSpinBox:focus, QDoubleSpinBox:focus {{
        border: 2px solid {palette.border_focus};
        padding: 6px 9px;
        padding-right: 27px;
    }}

    QSpinBox:disabled, QDoubleSpinBox:disabled {{
        background-color: {palette.bg_secondary};
        color: {palette.text_disabled};
    }}

    QSpinBox::up-button, QDoubleSpinBox::up-button {{
        background-color: {palette.bg_secondary};
        border: none;
        border-left: 1px solid {palette.border_normal};
        border-top-right-radius: 4px;
        width: 24px;
        subcontrol-position: top right;
    }}

    QSpinBox::down-button, QDoubleSpinBox::down-button {{
        background-color: {palette.bg_secondary};
        border: none;
        border-left: 1px solid {palette.border_normal};
        border-bottom-right-radius: 4px;
        width: 24px;
        subcontrol-position: bottom right;
    }}

    QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover {{
        background-color: {palette.bg_hover};
    }}

    QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{
        background-color: {palette.bg_hover};
    }}

    QSpinBox::up-button:pressed, QDoubleSpinBox::up-button:pressed {{
        background-color: {palette.border_normal};
    }}

    QSpinBox::down-button:pressed, QDoubleSpinBox::down-button:pressed {{
        background-color: {palette.border_normal};
    }}

    QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{
        image: none;
        border-left: 5px solid transparent;
        border-right: 5px solid transparent;
        border-bottom: 7px solid {palette.text_primary};
        width: 10px;
        height: 7px;
        margin: 2px;
    }}

    QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{
        image: none;
        border-left: 5px solid transparent;
        border-right: 5px solid transparent;
        border-top: 7px solid {palette.text_primary};
        width: 10px;
        height: 7px;
        margin: 2px;
    }}

    QSpinBox::up-arrow:hover, QDoubleSpinBox::up-arrow:hover {{
        border-bottom-color: {palette.accent_primary};
    }}

    QSpinBox::down-arrow:hover, QDoubleSpinBox::down-arrow:hover {{
        border-top-color: {palette.accent_primary};
    }}

    QSpinBox::up-arrow:disabled, QDoubleSpinBox::up-arrow:disabled {{
        border-bottom-color: {palette.text_disabled};
    }}

    QSpinBox::down-arrow:disabled, QDoubleSpinBox::down-arrow:disabled {{
        border-top-color: {palette.text_disabled};
    }}

    /* ===== Check Boxes ===== */
    QCheckBox {{
        spacing: 8px;
        color: {palette.text_primary};
    }}

    QCheckBox::indicator {{
        width: 18px;
        height: 18px;
        border: 1px solid {palette.border_normal};
        border-radius: 4px;
        background-color: {palette.bg_tertiary};
    }}

    QCheckBox::indicator:hover {{
        border-color: {palette.border_hover};
    }}

    QCheckBox::indicator:checked {{
        background-color: {palette.accent_primary};
        border-color: {palette.accent_primary};
        image: none;
    }}

    QCheckBox::indicator:checked:hover {{
        background-color: {palette.accent_hover};
        border-color: {palette.accent_hover};
    }}

    QCheckBox::indicator:disabled {{
        background-color: {palette.bg_secondary};
        border-color: {palette.border_normal};
    }}

    /* ===== Radio Buttons ===== */
    QRadioButton {{
        spacing: 8px;
        color: {palette.text_primary};
    }}

    QRadioButton::indicator {{
        width: 18px;
        height: 18px;
        border: 1px solid {palette.border_normal};
        border-radius: 9px;
        background-color: {palette.bg_tertiary};
    }}

    QRadioButton::indicator:hover {{
        border-color: {palette.border_hover};
    }}

    QRadioButton::indicator:checked {{
        background-color: {palette.accent_primary};
        border-color: {palette.accent_primary};
        border-width: 5px;
    }}

    QRadioButton::indicator:checked:hover {{
        background-color: {palette.accent_hover};
        border-color: {palette.accent_hover};
    }}

    QRadioButton::indicator:disabled {{
        background-color: {palette.bg_secondary};
        border-color: {palette.border_normal};
    }}

    /* ===== Progress Bars ===== */
    QProgressBar {{
        background-color: {palette.bg_tertiary};
        border: 1px solid {palette.border_normal};
        border-radius: 5px;
        text-align: center;
        color: {palette.text_primary};
        height: 24px;
    }}

    QProgressBar::chunk {{
        background-color: {palette.accent_primary};
        border-radius: 4px;
    }}

    /* ===== Tables ===== */
    QTableWidget {{
        background-color: {palette.bg_tertiary};
        alternate-background-color: {palette.bg_secondary};
        gridline-color: {palette.border_normal};
        border: 1px solid {palette.border_normal};
        border-radius: 5px;
        selection-background-color: {palette.accent_primary};
        selection-color: white;
    }}

    QTableWidget::item {{
        padding: 8px;
        border: none;
    }}

    QTableWidget::item:hover {{
        background-color: {palette.bg_hover};
    }}

    QHeaderView::section {{
        background-color: {palette.bg_secondary};
        color: {palette.text_primary};
        padding: 10px;
        border: none;
        border-bottom: 2px solid {palette.border_normal};
        border-right: 1px solid {palette.border_normal};
        font-weight: 600;
    }}

    QHeaderView::section:hover {{
        background-color: {palette.bg_hover};
    }}

    /* ===== Tab Widget ===== */
    QTabWidget::pane {{
        border: 1px solid {palette.border_normal};
        border-radius: 5px;
        background-color: {palette.bg_primary};
        top: -1px;
    }}

    QTabBar::tab {{
        background-color: {palette.bg_secondary};
        color: {palette.text_secondary};
        border: 1px solid {palette.border_normal};
        border-bottom: none;
        border-top-left-radius: 5px;
        border-top-right-radius: 5px;
        padding: 10px 20px;
        margin-right: 2px;
        font-weight: 500;
    }}

    QTabBar::tab:hover {{
        background-color: {palette.bg_hover};
        color: {palette.text_primary};
    }}

    QTabBar::tab:selected {{
        background-color: {palette.bg_primary};
        color: {palette.text_primary};
        border-bottom: 2px solid {palette.accent_primary};
        font-weight: 600;
    }}

    /* ===== Scroll Bars ===== */
    QScrollBar:vertical {{
        background-color: {palette.bg_secondary};
        width: 12px;
        border-radius: 6px;
        margin: 0;
    }}

    QScrollBar::handle:vertical {{
        background-color: {palette.border_normal};
        border-radius: 6px;
        min-height: 30px;
    }}

    QScrollBar::handle:vertical:hover {{
        background-color: {palette.border_hover};
    }}

    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
        height: 0;
        border: none;
    }}

    QScrollBar:horizontal {{
        background-color: {palette.bg_secondary};
        height: 12px;
        border-radius: 6px;
        margin: 0;
    }}

    QScrollBar::handle:horizontal {{
        background-color: {palette.border_normal};
        border-radius: 6px;
        min-width: 30px;
    }}

    QScrollBar::handle:horizontal:hover {{
        background-color: {palette.border_hover};
    }}

    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
        width: 0;
        border: none;
    }}

    /* ===== Status Bar ===== */
    QStatusBar {{
        background-color: {palette.bg_secondary};
        border-top: 1px solid {palette.border_normal};
        color: {palette.text_secondary};
        padding: 6px 12px;
    }}

    QStatusBar::item {{
        border: none;
    }}

    /* ===== Menu Bar ===== */
    QMenuBar {{
        background-color: {palette.bg_secondary};
        color: {palette.text_primary};
        border-bottom: 1px solid {palette.border_normal};
        padding: 4px;
    }}

    QMenuBar::item {{
        background-color: transparent;
        padding: 6px 12px;
        border-radius: 4px;
    }}

    QMenuBar::item:selected {{
        background-color: {palette.bg_hover};
    }}

    QMenuBar::item:pressed {{
        background-color: {palette.accent_primary};
        color: white;
    }}

    /* ===== Menus ===== */
    QMenu {{
        background-color: {palette.bg_tertiary};
        border: 1px solid {palette.border_normal};
        border-radius: 5px;
        padding: 4px;
    }}

    QMenu::item {{
        padding: 8px 24px 8px 12px;
        border-radius: 3px;
    }}

    QMenu::item:selected {{
        background-color: {palette.accent_primary};
        color: white;
    }}

    QMenu::separator {{
        height: 1px;
        background-color: {palette.border_normal};
        margin: 4px 8px;
    }}

    /* ===== Splitter ===== */
    QSplitter::handle {{
        background-color: {palette.border_normal};
    }}

    QSplitter::handle:horizontal {{
        width: 2px;
    }}

    QSplitter::handle:vertical {{
        height: 2px;
    }}

    QSplitter::handle:hover {{
        background-color: {palette.accent_primary};
    }}

    /* Hide locked splitter handles */
    QSplitter#interactiveMainSplitter::handle {{
        background-color: transparent;
        width: 0px;
    }}

    /* ===== Tool Tips ===== */
    QToolTip {{
        background-color: {palette.bg_tertiary};
        color: {palette.text_primary};
        border: 1px solid {palette.border_normal};
        border-radius: 5px;
        padding: 6px 10px;
    }}
    """


# Convenience exports
DARK_STYLESHEET = generate_stylesheet(DARK_PALETTE)
LIGHT_STYLESHEET = generate_stylesheet(LIGHT_PALETTE)
