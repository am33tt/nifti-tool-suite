"""Open a single plot full size, still interactive.

Panels draw several plots into one small figure, which is too small to read
values off. Clicking a plot opens that plot alone in its own window, as a
live matplotlib figure with the navigation toolbar, so it can be panned,
zoomed and saved at any resolution.

Artists belong to the figure they were created in and an axes cannot be
moved between figures, so the figure is cloned through a pickle round-trip
(which copies the data, not a rendering) and every axes except the clicked
one is removed from the clone. The panel keeps its own canvas throughout.

Twin axes are kept together, since a curve and its second y-axis share one
rectangle. If a figure cannot be pickled, the window falls back to a
zoomable high-resolution image of the same plot.
"""

from __future__ import annotations

import io
import pickle

from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QDialog, QFileDialog, QGraphicsScene, QGraphicsView, QHBoxLayout, QLabel,
    QMessageBox, QVBoxLayout, QWidget,
)

from ..config import BG, BORDER, PANEL2, TEXT, TEXT_DIM
from .widgets import style_nav_toolbar, styled_btn

#: Size of the enlarged figure, in inches.
ENLARGED_SIZE_IN = (11.0, 7.5)
#: Rectangle the kept axes is given in the enlarged figure.
ENLARGED_RECT = (0.085, 0.10, 0.885, 0.83)
#: Resolution multiplier for the image fallback.
FALLBACK_DPI_SCALE = 2.4
#: Zoom limits of the image fallback, relative to the fitted size.
MIN_SCALE, MAX_SCALE = 0.2, 40.0
#: A press and release further apart than this is a drag, not a click.
CLICK_SLOP_PX = 4.0


# --- picking the clicked plot out of a figure ------------------------------

def axes_group_indices(fig, target) -> list:
    """Indices of *target* and any twin sharing its rectangle.

    Twins are separate axes drawn on top of each other, so a click reports
    whichever is on top. Returning the group keeps a curve and its second
    y-axis together.
    """
    axes = list(fig.axes)
    if target not in axes:
        return list(range(len(axes)))
    box = target.get_position()
    group = []
    for i, ax in enumerate(axes):
        other = ax.get_position()
        if (abs(other.x0 - box.x0) < 1e-6 and abs(other.y0 - box.y0) < 1e-6
                and abs(other.width - box.width) < 1e-6
                and abs(other.height - box.height) < 1e-6):
            group.append(i)
    return group or [axes.index(target)]


def _enlarge_text(ax):
    """Scale the fonts up for a figure that is now several times larger."""
    ax.title.set_fontsize(13)
    ax.xaxis.label.set_fontsize(11)
    ax.yaxis.label.set_fontsize(11)
    ax.tick_params(labelsize=10)
    legend = ax.get_legend()
    if legend is not None:
        for text in legend.get_texts():
            text.set_fontsize(9)


def single_plot_figure(fig, keep_indices):
    """Clone *fig* keeping only the axes named by *keep_indices*.

    Raises whatever pickling raises, so the caller can fall back.
    """
    clone = pickle.loads(pickle.dumps(fig))
    axes = list(clone.axes)
    keep = [axes[i] for i in keep_indices if 0 <= i < len(axes)]
    if not keep:
        return clone
    for ax in axes:
        if ax not in keep:
            clone.delaxes(ax)
    for ax in keep:
        ax.set_position(list(ENLARGED_RECT))
        _enlarge_text(ax)
    # The figure title describes the whole panel, so it is used only when
    # the plot has no title of its own.
    suptitle = getattr(clone, "_suptitle", None)
    if suptitle is not None and keep[0].get_title():
        suptitle.set_text("")
    clone.set_size_inches(*ENLARGED_SIZE_IN)
    return clone


def plot_title(fig, keep_indices, default: str = "Plot") -> str:
    """A name for the window, taken from the plot itself."""
    axes = list(fig.axes)
    for i in keep_indices:
        if 0 <= i < len(axes):
            title = axes[i].get_title()
            if title:
                return title
    suptitle = getattr(fig, "_suptitle", None)
    if suptitle is not None and suptitle.get_text():
        return suptitle.get_text()
    return default


# --- the live window -------------------------------------------------------

class LivePlotDialog(QDialog):
    """One plot, full size, with the matplotlib navigation toolbar."""

    def __init__(self, parent, figure, title: str):
        super().__init__(parent)
        from matplotlib.backends.backend_qtagg import (
            FigureCanvasQTAgg, NavigationToolbar2QT,
        )

        self.setWindowTitle(title)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setStyleSheet(f"background-color: {BG}; color: {TEXT};")
        self.resize(1120, 780)

        self._figure = figure
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        self._canvas = FigureCanvasQTAgg(figure)
        root.addWidget(self._canvas, 1)

        bar = QWidget(self)
        bar_lay = QHBoxLayout(bar)
        bar_lay.setContentsMargins(0, 0, 0, 0)
        bar_lay.setSpacing(6)
        toolbar = style_nav_toolbar(NavigationToolbar2QT(self._canvas, bar))
        bar_lay.addWidget(toolbar)
        bar_lay.addStretch(1)
        bar_lay.addWidget(styled_btn(bar, "Close", self.close, small=True))
        root.addWidget(bar)

        self._canvas.draw_idle()


class ImagePlotDialog(QDialog):
    """Fallback window: a zoomable high-resolution image of the plot."""

    def __init__(self, parent, pixmap: QPixmap, title: str):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setStyleSheet(f"background-color: {BG}; color: {TEXT};")
        self.resize(1120, 780)

        self._pixmap = pixmap
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(6)

        self._view = ZoomableImageView(pixmap, self)
        root.addWidget(self._view, 1)

        bar = QWidget(self)
        bar_lay = QHBoxLayout(bar)
        bar_lay.setContentsMargins(0, 0, 0, 0)
        bar_lay.setSpacing(6)
        hint = QLabel("Scroll to zoom  ·  drag to pan  ·  double-click to fit",
                      bar)
        hint.setStyleSheet(
            f"color: {TEXT_DIM}; background-color: {PANEL2}; "
            f"border: 1px solid {BORDER}; padding: 4px 8px;")
        bar_lay.addWidget(hint, 1)
        bar_lay.addWidget(styled_btn(bar, "Fit", self._view.fit, small=True))
        bar_lay.addWidget(styled_btn(bar, "Save image...", self._save,
                                     small=True))
        bar_lay.addWidget(styled_btn(bar, "Close", self.close, small=True))
        root.addWidget(bar)

    def _save(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save image", "plot.png",
            "PNG image (*.png);;All files (*.*)")
        if not path:
            return
        if not self._pixmap.save(path):
            QMessageBox.warning(self, "Could not save",
                                f"Writing {path} failed.")


class ZoomableImageView(QGraphicsView):
    """Image view with wheel zoom, drag pan and fit-to-window."""

    def __init__(self, pixmap: QPixmap, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self._item = self._scene.addPixmap(pixmap)
        self.setScene(self._scene)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(
            QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setBackgroundBrush(Qt.GlobalColor.white)
        self.setFrameShape(QGraphicsView.Shape.NoFrame)
        self._fitted = False
        self._scale = 1.0

    def fit(self):
        rect = self._item.boundingRect()
        if rect.isEmpty():
            return
        self.fitInView(rect, Qt.AspectRatioMode.KeepAspectRatio)
        self._scale = 1.0
        self._fitted = True

    def showEvent(self, event):
        super().showEvent(event)
        if not self._fitted:
            self.fit()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if abs(self._scale - 1.0) < 1e-6:
            self.fit()

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        if delta == 0:
            return
        factor = 1.25 if delta > 0 else 1 / 1.25
        new_scale = self._scale * factor
        if not (MIN_SCALE <= new_scale <= MAX_SCALE):
            return
        self._scale = new_scale
        self.scale(factor, factor)

    def mouseDoubleClickEvent(self, event):
        self.fit()


def pixmap_from_figure(fig, scale: float = FALLBACK_DPI_SCALE) -> QPixmap:
    """Render a matplotlib figure to a pixmap at a raised resolution."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=fig.get_dpi() * float(scale),
                facecolor=fig.get_facecolor(), bbox_inches="tight")
    buf.seek(0)
    return QPixmap.fromImage(QImage.fromData(buf.getvalue(), "PNG"))


# --- entry points ----------------------------------------------------------

def show_plot(parent, fig, keep_indices=None, title: str | None = None):
    """Open one plot of *fig* in its own window.

    ``keep_indices`` selects which axes to show; ``None`` shows all of them,
    used only when the figure holds a single plot.
    """
    if keep_indices is None:
        keep_indices = list(range(len(fig.axes)))
    window_title = title or plot_title(fig, keep_indices)
    try:
        clone = single_plot_figure(fig, keep_indices)
        dialog = LivePlotDialog(parent, clone, window_title)
    except Exception:
        # Pickling or the clone failed; show the plot as an image instead.
        try:
            dialog = ImagePlotDialog(parent, pixmap_from_figure(fig),
                                     window_title)
        except Exception:
            return None
    dialog.show()
    dialog.raise_()
    dialog.activateWindow()
    return dialog


def enable_click_to_enlarge(canvas, fig, parent, name: str = "Plot"):
    """Open the plot under the pointer when its panel is clicked.

    A click inside one of the figure's axes opens that axes alone. A click
    on the figure background is ignored, since there is no single plot to
    show.
    """
    state = {"x": None, "y": None}

    def _on_press(event):
        state["x"], state["y"] = event.x, event.y

    def _on_release(event):
        if getattr(event, "button", 1) != 1:
            return
        # Ignore the release that ends a pan or a zoom rectangle.
        toolbar = getattr(canvas, "toolbar", None)
        if toolbar is not None and getattr(toolbar, "mode", ""):
            return
        # Ignore a drag: only a click opens the plot.
        if state["x"] is not None and event.x is not None:
            if (abs(event.x - state["x"]) > CLICK_SLOP_PX
                    or abs(event.y - state["y"]) > CLICK_SLOP_PX):
                return
        target = getattr(event, "inaxes", None)
        if target is None:
            return
        try:
            show_plot(parent, fig, axes_group_indices(fig, target))
        except Exception:
            pass

    canvas.mpl_connect("button_press_event", _on_press)
    canvas.mpl_connect("button_release_event", _on_release)
    try:
        canvas.setCursor(Qt.CursorShape.PointingHandCursor)
    except Exception:
        pass
    return canvas


def click_hint(parent) -> QLabel:
    """Hint label telling the user the plots are clickable."""
    label = QLabel("Click a plot to open it on its own, full size, with "
                   "pan, zoom and save.", parent)
    label.setStyleSheet(
        f"color: {TEXT_DIM}; background-color: transparent; padding: 0 8px;")
    return label
