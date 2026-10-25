
import math
import time
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox

import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange
from matplotlib.widgets import RectangleSelector


# ============================================================
# POWER TOWER FRACTAL EXPLORER
# z_(n+1) = c ** z_n = exp(z_n * Log(c))
#
# Features:
# - Black and white only
# - Mouse wheel zoom
# - Drag rectangle zoom
# - Deep repeated zoom (float64 precision limits apply)
# - Hover coordinate display
# - Right-click to copy complex coordinate
# - S: export current view as high-resolution PNG
# - R: reset view
# - Q / Esc: quit
# ============================================================

# ---------- Interactive preview settings ----------
PREVIEW_W = 1200
PREVIEW_H = 750
PREVIEW_ITER = 250

# ---------- High-resolution export defaults ----------
EXPORT_W = 3000
EXPORT_H = 1600
EXPORT_ITER = 500

ESCAPE_RADIUS = 1.0e6
WHEEL_FACTOR = 1.35
REFINE_DELAY_MS = 650

INITIAL_X = (-2.0, 2.0)
INITIAL_Y = (-4.0 / 3.0, 4.0 / 3.0)

# Avoid zooming past useful double-precision resolution
MIN_SPAN = 1.0e-14


@njit(parallel=True, cache=True)
def calculate_fractal(xmin, xmax, ymin, ymax, width, height, max_iter):
    """Return uint8 image: 0=black, 255=white."""

    image = np.zeros((height, width), dtype=np.uint8)
    escape_sq = ESCAPE_RADIUS * ESCAPE_RADIUS
    log_escape = math.log(ESCAPE_RADIUS)

    for j in prange(height):
        ci = ymin + (ymax - ymin) * j / max(1, height - 1)

        for i in range(width):
            cr = xmin + (xmax - xmin) * i / max(1, width - 1)

            # Log(0) is undefined
            if cr == 0.0 and ci == 0.0:
                continue

            log_r = 0.5 * math.log(cr * cr + ci * ci)
            theta = math.atan2(ci, cr)

            zr = 0.0
            zi = 0.0
            escaped = False

            for _ in range(max_iter):
                # w = z * Log(c)
                wr = zr * log_r - zi * theta
                wi = zr * theta + zi * log_r

                if not math.isfinite(wr) or not math.isfinite(wi):
                    escaped = True
                    break

                if wr > log_escape:
                    escaped = True
                    break

                if wr < -745.0:
                    zr = 0.0
                    zi = 0.0
                else:
                    scale = math.exp(wr)
                    zr = scale * math.cos(wi)
                    zi = scale * math.sin(wi)

                if not math.isfinite(zr) or not math.isfinite(zi):
                    escaped = True
                    break

                if zr * zr + zi * zi > escape_sq:
                    escaped = True
                    break

            if escaped:
                image[j, i] = 255

    return image


class PowerTowerExplorer:
    def __init__(self):
        self.xlim = list(INITIAL_X)
        self.ylim = list(INITIAL_Y)

        self.busy = False
        self.last_change = time.perf_counter()
        self.last_mouse_complex = None

        self.root = tk.Tk()
        self.root.withdraw()

        # Full-window image: no axes, ticks, title, or colorbar
        self.fig = plt.figure(figsize=(12, 8), facecolor="black")
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_axis_off()
        self.ax.set_facecolor("black")

        try:
            self.fig.canvas.manager.set_window_title(
                "Power Tower Fractal | S: Save PNG | Right-click: Copy coordinate"
            )
        except Exception:
            pass

        try:
            manager = self.fig.canvas.manager
            window = getattr(manager, "window", None)
            if window is not None and hasattr(window, "state"):
                window.state("zoomed")
            elif window is not None and hasattr(window, "showMaximized"):
                window.showMaximized()
        except Exception:
            pass

        self.image = self.ax.imshow(
            np.zeros((PREVIEW_H, PREVIEW_W), dtype=np.uint8),
            extent=(
                self.xlim[0], self.xlim[1],
                self.ylim[0], self.ylim[1],
            ),
            origin="lower",
            cmap="gray",
            vmin=0,
            vmax=255,
            interpolation="nearest",
            aspect="auto",
        )

        # Small coordinate overlay; the graph has no axes or colorbar.
        self.hud = self.ax.text(
            0.012, 0.018, "",
            transform=self.ax.transAxes,
            color="white",
            fontsize=9,
            family="monospace",
            verticalalignment="bottom",
            horizontalalignment="left",
            bbox=dict(facecolor="black", alpha=0.65, edgecolor="none"),
        )

        self.selector = RectangleSelector(
            self.ax,
            self.on_select,
            button=[1],
            minspanx=5,
            minspany=5,
            spancoords="pixels",
            interactive=False,
            useblit=False,
            props={
                "facecolor": "white",
                "edgecolor": "white",
                "alpha": 0.20,
                "fill": True,
            },
        )

        self.fig.canvas.mpl_connect("scroll_event", self.on_scroll)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.fig.canvas.mpl_connect("button_press_event", self.on_mouse_press)

        self.refine_timer = self.fig.canvas.new_timer(
            interval=REFINE_DELAY_MS
        )
        self.refine_timer.add_callback(self.refine_if_idle)

        self.render_preview()

    def update_extent(self):
        self.image.set_extent((
            self.xlim[0], self.xlim[1],
            self.ylim[0], self.ylim[1],
        ))
        self.ax.set_xlim(self.xlim)
        self.ax.set_ylim(self.ylim)

    def compute(self, width, height, iterations, update_screen=True):
        start = time.perf_counter()

        data = calculate_fractal(
            self.xlim[0], self.xlim[1],
            self.ylim[0], self.ylim[1],
            width, height, iterations,
        )

        # Force exact binary black/white values
        data = np.where(data >= 128, 255, 0).astype(np.uint8)

        if update_screen:
            self.image.set_data(data)
            self.update_extent()
            self.fig.canvas.draw()

        elapsed = time.perf_counter() - start
        print(
            f"{width} x {height}, iterations={iterations}: "
            f"{elapsed:.2f} seconds"
        )
        return data

    def render_preview(self):
        if self.busy:
            return

        self.busy = True
        try:
            self.compute(
                PREVIEW_W, PREVIEW_H, PREVIEW_ITER,
                update_screen=True,
            )
        finally:
            self.busy = False

        self.last_change = time.perf_counter()
        self.refine_timer.stop()
        self.refine_timer.start()

    def refine_if_idle(self):
        self.refine_timer.stop()

        if self.busy:
            return

        elapsed_ms = (time.perf_counter() - self.last_change) * 1000
        if elapsed_ms < REFINE_DELAY_MS:
            self.refine_timer.start()
            return

        # Recalculate at a slightly higher quality after interaction stops.
        self.busy = True
        try:
            self.compute(
                1600, 1000, 300,
                update_screen=True,
            )
        finally:
            self.busy = False

    def changed_view(self):
        self.last_change = time.perf_counter()
        self.render_preview()

    def zoom_at(self, x, y, factor):
        if x is None or y is None or self.busy:
            return

        x0, x1 = self.xlim
        y0, y1 = self.ylim

        old_w = x1 - x0
        old_h = y1 - y0
        new_w = old_w / factor
        new_h = old_h / factor

        if new_w < MIN_SPAN or new_h < MIN_SPAN:
            print("일반 double-precision 계산의 정밀도 한계에 가까워졌습니다.")
            return

        rx = (x - x0) / old_w
        ry = (y - y0) / old_h

        self.xlim = [
            x - rx * new_w,
            x + (1 - rx) * new_w,
        ]
        self.ylim = [
            y - ry * new_h,
            y + (1 - ry) * new_h,
        ]

        self.changed_view()

    def on_scroll(self, event):
        if event.inaxes != self.ax or self.busy:
            return

        if event.button == "up":
            self.zoom_at(event.xdata, event.ydata, WHEEL_FACTOR)
        elif event.button == "down":
            self.zoom_at(event.xdata, event.ydata, 1 / WHEEL_FACTOR)

    def on_select(self, press, release):
        if self.busy:
            return

        if (
            press.xdata is None or press.ydata is None
            or release.xdata is None or release.ydata is None
        ):
            return

        x0, x1 = sorted([press.xdata, release.xdata])
        y0, y1 = sorted([press.ydata, release.ydata])

        if x1 - x0 < MIN_SPAN or y1 - y0 < MIN_SPAN:
            return

        cx = (x0 + x1) / 2
        cy = (y0 + y1) / 2

        ratio = self.ax.bbox.width / max(self.ax.bbox.height, 1)
        w = x1 - x0
        h = y1 - y0

        if w / h > ratio:
            h = w / ratio
        else:
            w = h * ratio

        if w < MIN_SPAN or h < MIN_SPAN:
            print("선택한 영역이 double-precision 정밀도 한계보다 작습니다.")
            return

        self.xlim = [cx - w / 2, cx + w / 2]
        self.ylim = [cy - h / 2, cy + h / 2]

        self.changed_view()

    def on_motion(self, event):
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return

        x = event.xdata
        y = event.ydata
        self.last_mouse_complex = (x, y)

        self.hud.set_text(
            f"c = {x:.12g} {'+' if y >= 0 else '-'} {abs(y):.12g}i"
            f"\nRe: {x:.12g}   Im: {y:.12g}"
            f"\nWidth: {self.xlim[1] - self.xlim[0]:.5g}"
        )
        self.fig.canvas.draw_idle()

    def on_mouse_press(self, event):
        # Right-click copies the complex coordinate under the pointer.
        if event.button == 3 and event.inaxes == self.ax:
            if event.xdata is None or event.ydata is None:
                return

            x = event.xdata
            y = event.ydata
            coord = f"{x:.16g}{'+' if y >= 0 else '-'}{abs(y):.16g}j"

            try:
                self.root.clipboard_clear()
                self.root.clipboard_append(coord)
                self.root.update()
                print(f"클립보드에 복사됨: {coord}")
                self.hud.set_text(f"Copied: {coord}")
                self.fig.canvas.draw_idle()
            except Exception as exc:
                print(f"복사 실패: {exc}")
                print(f"좌표: {coord}")

    def save_high_resolution(self):
        if self.busy:
            print("현재 계산이 끝난 뒤 저장해 주세요.")
            return

        filename = filedialog.asksaveasfilename(
            parent=self.root,
            title="프랙털 PNG 저장",
            defaultextension=".png",
            filetypes=[("PNG image", "*.png")],
            initialfile="power_tower_fractal.png",
        )
        if not filename:
            return

        width = simpledialog.askinteger(
            "고해상도 저장",
            "이미지 너비 (픽셀):",
            initialvalue=EXPORT_W,
            minvalue=200,
            maxvalue=12000,
            parent=self.root,
        )
        if width is None:
            return

        height = simpledialog.askinteger(
            "고해상도 저장",
            "이미지 높이 (픽셀):",
            initialvalue=EXPORT_H,
            minvalue=200,
            maxvalue=12000,
            parent=self.root,
        )
        if height is None:
            return

        iterations = simpledialog.askinteger(
            "고해상도 저장",
            "최대 반복 횟수:",
            initialvalue=EXPORT_ITER,
            minvalue=10,
            maxvalue=5000,
            parent=self.root,
        )
        if iterations is None:
            return

        self.busy = True
        try:
            print("고해상도 이미지 계산 중...")
            data = self.compute(
                width, height, iterations,
                update_screen=False,
            )

            # Save a pure black-and-white PNG with no axes or margins.
            plt.imsave(
                filename,
                data,
                cmap="gray",
                vmin=0,
                vmax=255,
                format="png",
            )
            print(f"저장 완료: {filename}")
            messagebox.showinfo(
                "저장 완료",
                f"PNG 이미지 저장 완료!\n{filename}\n"
                f"{width} x {height} pixels",
                parent=self.root,
            )
        except Exception as exc:
            messagebox.showerror(
                "저장 오류", str(exc), parent=self.root
            )
        finally:
            self.busy = False

    def on_key(self, event):
        if event.key in ("r", "R"):
            self.xlim = list(INITIAL_X)
            self.ylim = list(INITIAL_Y)
            self.changed_view()

        elif event.key in ("s", "S"):
            self.save_high_resolution()

        elif event.key in ("+", "=", "plus"):
            self.zoom_at(
                (self.xlim[0] + self.xlim[1]) / 2,
                (self.ylim[0] + self.ylim[1]) / 2,
                WHEEL_FACTOR,
            )

        elif event.key in ("-", "minus"):
            self.zoom_at(
                (self.xlim[0] + self.xlim[1]) / 2,
                (self.ylim[0] + self.ylim[1]) / 2,
                1 / WHEEL_FACTOR,
            )

        elif event.key in ("q", "escape"):
            self.refine_timer.stop()
            plt.close(self.fig)

    def show(self):
        plt.show()


if __name__ == "__main__":
    print("=" * 55)
    print("POWER TOWER FRACTAL EXPLORER")
    print("z_(n+1) = c ** z_n")
    print("=" * 55)
    print("마우스 휠 위/아래 : 확대/축소")
    print("왼쪽 드래그       : 영역 확대")
    print("마우스 오른쪽 클릭: 복소수 좌표 복사")
    print("마우스 이동       : 좌표 확인")
    print("S                 : 고해상도 PNG 저장")
    print("R                 : 초기 화면")
    print("+ / -             : 화면 중앙 확대/축소")
    print("Q 또는 Esc        : 종료")
    print("=" * 55)

    app = PowerTowerExplorer()
    app.show()
