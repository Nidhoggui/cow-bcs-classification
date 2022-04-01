from tkinter import *
from tkinter import filedialog
from PIL import Image
from PIL import ImageTk

import cv2
import numpy as np
from skimage.segmentation import slic, mark_boundaries
from skimage.util import img_as_float


class SuperpixelApp(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)

        self.image_openCV = None
        self.image_TK = None
        self.result_image_openCV = None
        self.result_image_TK = None
        self.mask = None
        self.segments_slic = None

        self.canvas = None
        self.canvas_result = None

        self.init_UI()

    def init_UI(self):
        self.master.title("Superpixels")
        self.pack()

        self.master.bind("<ButtonPress-1>", self.click_event)

        self.image_TK, self.result_image_TK = self.load_image()

        self.create_result_window()

        self.canvas = Canvas(self.master, width=self.image_TK.width(), height=self.image_TK.height())
        self.canvas.create_image(0, 0, anchor=NW, image=self.image_TK)
        self.canvas.image = self.image_TK
        self.canvas.pack()

    def create_result_window(self):
        window = Toplevel(self.master) # Toplevel is a window child of Frame
        window.wm_title("Result")
        window.minsize(width=self.result_image_TK.width(), height=self.result_image_TK.height())
        
        self.canvas_result = Canvas(window, width=self.result_image_TK.width(), height=self.result_image_TK.height())
        self.canvas_result.create_image(0, 0, anchor=NW, image=self.result_image_TK)
        self.canvas_result.pack()


    def click_event(self, event):
        x = int(self.canvas.canvasx(event.x))
        y = int(self.canvas.canvasy(event.y))

        clicked_label = self.segments_slic[y][x]
        self.mask[self.segments_slic == clicked_label] = 1

        self.result_image_openCV = self.image_openCV * self.mask[:, :, np.newaxis]

        result_image_PIL = Image.fromarray(self.result_image_openCV)
        self.result_image_TK = ImageTk.PhotoImage(result_image_PIL)
        self.canvas_result.create_image(0, 0, anchor=NW, image=self.result_image_TK)
        self.canvas_result.pack()


    def load_image(self):
        image_path = filedialog.askopenfilename()

        if (len(image_path) > 0):
            self.image_openCV = cv2.imread(image_path)
            self.image_openCV = cv2.cvtColor(self.image_openCV, cv2.COLOR_BGR2RGB)

            # calculate slic and mark the superpixels
            self.segments_slic = slic(img_as_float(self.image_openCV), n_segments=150, compactness=20, sigma=1, start_label=0)
            image_boundaries = mark_boundaries(img_as_float(self.image_openCV), self.segments_slic, color=(255, 254, 253))

            image_boundaries_openCV = self.image_openCV.copy()
            image_boundaries_openCV[image_boundaries == 255] = 255
            image_boundaries_openCV[image_boundaries == 254] = 0
            image_boundaries_openCV[image_boundaries == 253] = 0

            self.mask = np.zeros(self.image_openCV.shape[:2], np.uint8)
            self.result_image_openCV = np.zeros(self.image_openCV.shape, np.uint8)

            # Pillow format image
            image_PIL = Image.fromarray(image_boundaries_openCV) 
            result_image_PIL = Image.fromarray(self.result_image_openCV)
            
            return ImageTk.PhotoImage(image_PIL), ImageTk.PhotoImage(result_image_PIL)

        return None, None
    

if __name__ == "__main__":
    root = Tk()

    app = SuperpixelApp(master=root)

    app.mainloop()
