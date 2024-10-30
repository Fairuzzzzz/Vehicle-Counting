import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import json
import cv2
import numpy as np

class LineDrawerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Multi-Line Drawing Tool for Vehicle Counting")

        # Initialize variables
        self.image_path = None
        self.image = None
        self.photo = None
        self.start_point = None
        self.end_point = None
        self.drawing = False
        self.current_line = None

        # List to store all lines
        self.lines = []  # Will store tuples of (start_point, end_point, canvas_line_id)

        # Create GUI elements
        self.create_widgets()

    def create_widgets(self):
        # Top frame for buttons
        self.button_frame = tk.Frame(self.root)
        self.button_frame.pack(pady=5)

        # Load Image Button
        self.load_btn = tk.Button(
            self.button_frame,
            text="Load Image",
            command=self.load_image
        )
        self.load_btn.pack(side=tk.LEFT, padx=5)

        # Save Coordinates Button
        self.save_btn = tk.Button(
            self.button_frame,
            text="Save All Lines",
            command=self.save_coordinates,
            state=tk.DISABLED
        )
        self.save_btn.pack(side=tk.LEFT, padx=5)

        # Clear Last Line Button
        self.clear_last_btn = tk.Button(
            self.button_frame,
            text="Clear Last Line",
            command=self.clear_last_line,
            state=tk.DISABLED
        )
        self.clear_last_btn.pack(side=tk.LEFT, padx=5)

        # Clear All Lines Button
        self.clear_all_btn = tk.Button(
            self.button_frame,
            text="Clear All Lines",
            command=self.clear_all_lines,
            state=tk.DISABLED
        )
        self.clear_all_btn.pack(side=tk.LEFT, padx=5)

        # Line counter label
        self.line_counter = tk.Label(
            self.button_frame,
            text="Lines: 0",
            padx=10
        )
        self.line_counter.pack(side=tk.LEFT)

        # Canvas for image and line drawing
        self.canvas = tk.Canvas(self.root)
        self.canvas.pack(expand=True, fill=tk.BOTH, padx=10, pady=5)

        # Instructions label
        self.instructions = tk.Label(
            self.root,
            text="Click and drag to draw lines. Each line will be saved and numbered. Use Clear Last Line to remove the most recent line.",
            wraplength=400
        )
        self.instructions.pack(pady=5)

        # Bind mouse events
        self.canvas.bind("<ButtonPress-1>", self.start_line)
        self.canvas.bind("<B1-Motion>", self.draw_line)
        self.canvas.bind("<ButtonRelease-1>", self.end_line)

    def load_image(self):
        self.image_path = filedialog.askopenfilename(
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff")
            ]
        )

        if self.image_path:
            # Load and display image
            self.image = Image.open(self.image_path)

            # Get window size
            window_width = self.root.winfo_width()
            window_height = self.root.winfo_height()

            # Calculate scaling factor
            image_ratio = self.image.width / self.image.height
            window_ratio = window_width / window_height

            if window_ratio > image_ratio:
                new_height = window_height - 100
                new_width = int(new_height * image_ratio)
            else:
                new_width = window_width - 20
                new_height = int(new_width / image_ratio)

            # Resize image
            self.image = self.image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            self.photo = ImageTk.PhotoImage(self.image)

            # Update canvas
            self.canvas.config(width=new_width, height=new_height)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

            # Enable buttons
            self.save_btn.config(state=tk.NORMAL)
            self.clear_last_btn.config(state=tk.NORMAL)
            self.clear_all_btn.config(state=tk.NORMAL)

    def start_line(self, event):
        self.drawing = True
        self.start_point = (event.x, event.y)

    def draw_line(self, event):
        if self.drawing and self.start_point:
            # Remove temporary line if it exists
            if self.current_line:
                self.canvas.delete(self.current_line)

            # Draw new temporary line
            self.current_line = self.canvas.create_line(
                self.start_point[0],
                self.start_point[1],
                event.x,
                event.y,
                fill="red",
                width=2
            )

    def end_line(self, event):
        if self.drawing and self.start_point:
            self.drawing = False
            self.end_point = (event.x, event.y)

            # Create permanent line
            line_id = self.canvas.create_line(
                self.start_point[0],
                self.start_point[1],
                self.end_point[0],
                self.end_point[1],
                fill="red",
                width=2
            )

            # Add line number label
            label_x = (self.start_point[0] + self.end_point[0]) // 2
            label_y = (self.start_point[1] + self.end_point[1]) // 2
            label_id = self.canvas.create_text(
                label_x,
                label_y - 10,
                text=str(len(self.lines) + 1),
                fill="blue",
                font=("Arial", 12, "bold")
            )

            # Store line information
            self.lines.append({
                'start': self.start_point,
                'end': self.end_point,
                'line_id': line_id,
                'label_id': label_id
            })

            # Reset current line
            self.current_line = None

            # Update line counter
            self.line_counter.config(text=f"Lines: {len(self.lines)}")

    def clear_last_line(self):
        if self.lines:
            # Get last line info
            last_line = self.lines.pop()

            # Remove line and label from canvas
            self.canvas.delete(last_line['line_id'])
            self.canvas.delete(last_line['label_id'])

            # Update line counter
            self.line_counter.config(text=f"Lines: {len(self.lines)}")

    def clear_all_lines(self):
        # Remove all lines and labels from canvas
        for line in self.lines:
            self.canvas.delete(line['line_id'])
            self.canvas.delete(line['label_id'])

        # Clear lines list
        self.lines.clear()

        # Update line counter
        self.line_counter.config(text="Lines: 0")

    def save_coordinates(self):
        if not self.lines:
            messagebox.showwarning(
                "Warning",
                "Please draw at least one line first!"
            )
            return

        if self.image_path:
            # Get original image dimensions
            original_img = cv2.imread(self.image_path)
            orig_height, orig_width = original_img.shape[:2]

            # Calculate scaling factors
            scale_x = orig_width / self.image.width
            scale_y = orig_height / self.image.height

            # Scale all line coordinates back to original image size
            lines_data = []
            for i, line in enumerate(self.lines, 1):
                original_start = (
                    int(line['start'][0] * scale_x),
                    int(line['start'][1] * scale_y)
                )
                original_end = (
                    int(line['end'][0] * scale_x),
                    int(line['end'][1] * scale_y)
                )

                lines_data.append({
                    'line_number': i,
                    'start_point': original_start,
                    'end_point': original_end
                })

            # Save coordinates to JSON file
            coordinates = {
                'lines': lines_data,
                'image_width': orig_width,
                'image_height': orig_height
            }

            save_path = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON files", "*.json")]
            )

            if save_path:
                with open(save_path, 'w') as f:
                    json.dump(coordinates, f, indent=2)
                messagebox.showinfo(
                    "Success",
                    f"All line coordinates saved to {save_path}"
                )

def main():
    root = tk.Tk()
    app = LineDrawerGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
