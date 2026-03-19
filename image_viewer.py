import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import os

# ── CONFIG ──────────────────────────────────────────────────────────────────
DIR_LEFT  = "/Users/jaykumarparekh/Documents/Research/drone_postprocessing/yolo_model_testing/non_xai_results/images"
DIR_RIGHT = "/Users/jaykumarparekh/Documents/Research/drone_postprocessing/yolo_model_testing/xai_results/postprocessed_images"
# ────────────────────────────────────────────────────────────────────────────

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tiff"}


def load_images(directory):
    files = sorted(
        f for f in os.listdir(directory)
        if os.path.splitext(f)[1].lower() in IMG_EXTS
    )
    if not files:
        raise ValueError(f"No images found in: {directory}")
    return [os.path.join(directory, f) for f in files]


class SideBySideViewer:
    def __init__(self, root, left_paths, right_paths):
        self.root = root
        self.root.title("Image Viewer")
        self.root.configure(bg="#1e1e1e")

        self.paths = [left_paths, right_paths]
        self.indices = [0, 0]
        self.photo_refs = [None, None]   # keep refs to avoid GC

        self._build_ui()
        self._bind_keys()
        self.root.update()
        self._refresh(0)
        self._refresh(1)

    # ── UI ──────────────────────────────────────────────────────────────────

    def _build_ui(self):
        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        self.panels = []
        for side, title in enumerate(["Left", "Right"]):
            frame = tk.Frame(self.root, bg="#1e1e1e", padx=6, pady=6)
            frame.grid(row=0, column=side, sticky="nsew")
            frame.columnconfigure(0, weight=1)
            frame.rowconfigure(1, weight=1)

            # filename label
            lbl_name = tk.Label(frame, text="", bg="#1e1e1e", fg="#cccccc",
                                font=("Helvetica", 9))
            lbl_name.grid(row=0, column=0, pady=(0, 4))

            # image canvas
            canvas = tk.Canvas(frame, bg="#2d2d2d", highlightthickness=0)
            canvas.grid(row=1, column=0, sticky="nsew")

            # counter + nav row
            nav = tk.Frame(frame, bg="#1e1e1e")
            nav.grid(row=2, column=0, pady=(4, 0))

            btn_prev = tk.Button(nav, text="◀  Prev", command=lambda s=side: self._step(s, -1),
                                 bg="#3a3a3a", fg="white", relief="flat",
                                 padx=10, pady=4, cursor="hand2")
            btn_prev.pack(side="left", padx=4)

            lbl_count = tk.Label(nav, text="", bg="#1e1e1e", fg="#aaaaaa",
                                 font=("Helvetica", 9), width=10, anchor="center")
            lbl_count.pack(side="left")

            btn_next = tk.Button(nav, text="Next  ▶", command=lambda s=side: self._step(s, +1),
                                 bg="#3a3a3a", fg="white", relief="flat",
                                 padx=10, pady=4, cursor="hand2")
            btn_next.pack(side="left", padx=4)

            self.panels.append({
                "frame": frame,
                "canvas": canvas,
                "lbl_name": lbl_name,
                "lbl_count": lbl_count,
            })

        # window resize → redraw
        self.root.bind("<Configure>", self._on_resize)
        self._resize_job = None

    def _bind_keys(self):
        # Left panel: A / D    Right panel: ← / →
        self.root.bind("<Left>",  lambda e: self._step(1, -1))
        self.root.bind("<Right>", lambda e: self._step(1, +1))
        self.root.bind("a",       lambda e: self._step(0, -1))
        self.root.bind("d",       lambda e: self._step(0, +1))
        self.root.bind("q",       lambda e: self.root.destroy())

    # ── Navigation ──────────────────────────────────────────────────────────

    def _step(self, side, delta):
        n = len(self.paths[side])
        self.indices[side] = (self.indices[side] + delta) % n
        self._refresh(side)

    # ── Rendering ───────────────────────────────────────────────────────────

    def _refresh(self, side):
        idx   = self.indices[side]
        path  = self.paths[side][idx]
        panel = self.panels[side]

        # labels
        panel["lbl_name"].config(text=os.path.basename(path))
        panel["lbl_count"].config(
            text=f"{idx + 1} / {len(self.paths[side])}"
        )

        # load & fit image
        canvas = panel["canvas"]
        cw = canvas.winfo_width()  or 600
        ch = canvas.winfo_height() or 600

        img = Image.open(path)
        img.thumbnail((cw, ch), Image.LANCZOS)
        photo = ImageTk.PhotoImage(img)

        self.photo_refs[side] = photo   # prevent GC
        canvas.delete("all")
        canvas.create_image(cw // 2, ch // 2, anchor="center", image=photo)

    def _on_resize(self, event):
        # debounce: redraw ~150 ms after last resize event
        if self._resize_job:
            self.root.after_cancel(self._resize_job)
        self._resize_job = self.root.after(150, self._redraw_all)

    def _redraw_all(self):
        self._refresh(0)
        self._refresh(1)


# ── Entry point ─────────────────────────────────────────────────────────────

def main():
    try:
        left_paths  = load_images(DIR_LEFT)
        right_paths = load_images(DIR_RIGHT)
    except (ValueError, FileNotFoundError) as e:
        print(f"Error: {e}")
        return

    root = tk.Tk()
    root.geometry("1400x800")
    root.minsize(800, 400)

    SideBySideViewer(root, left_paths, right_paths)

    print("Keys: A/D → left panel   |   ←/→ → right panel   |   Q → quit")
    root.mainloop()


if __name__ == "__main__":
    main()
