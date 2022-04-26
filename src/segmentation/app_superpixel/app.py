from tkinter import *
from tkinter import filedialog
from PIL import Image
from PIL import ImageTk

from cv2 import cv2
import numpy as np
from skimage.segmentation import slic, mark_boundaries
from skimage.util import img_as_float


class SuperpixelApp(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)

        # attributes
        self.image_path = ""
        self.image_openCV = None
        self.image_TK = None
        self.result_image_openCV = None
        self.result_image_TK = None
        self.mask = None
        self.segments_slic = None

        self.canvas = None
        self.padding_between_images = 10
        self.save_btn = None
        self.new_image_btn = None

        # methods
        self._init_UI()

    def _init_UI(self):
        self.master.title("Superpixels")
        self.pack()
        self._load_UI()

    def _load_UI(self):
        self.image_TK, self.result_image_TK = self._load_image()

        if self.image_TK is None or self.result_image_TK is None:
            self.master.destroy()
        else:
            self.canvas = Canvas(self.master, width=(self.image_TK.width() * 2) + self.padding_between_images,
                                 height=self.image_TK.height())
            self.canvas.create_image(0, 0, anchor=NW, image=self.image_TK)
            self.canvas.create_image(self.result_image_TK.width() + self.padding_between_images, 0, anchor=NW,
                                     image=self.result_image_TK)
            self.canvas.bind("<ButtonPress-1>", self._click_event)
            self.canvas.pack()

            self.save_btn = Button(self.master, text="Save result", command=self._save_result)
            self.save_btn.pack()
            self.new_image_btn = Button(self.master, text="Choose new image", command=self._load_new_image)
            self.new_image_btn.pack()

    def _load_new_image(self):
        self.canvas.destroy()
        self.save_btn.destroy()
        self.new_image_btn.destroy()
        self._load_UI()

    def _click_event(self, event):
        x = int(self.canvas.canvasx(event.x))
        y = int(self.canvas.canvasy(event.y))

        if y < self.segments_slic.shape[0] and x < self.segments_slic.shape[1]:
            clicked_label = self.segments_slic[y][x]
            self.mask[self.segments_slic == clicked_label] = 1
            self.result_image_openCV = self.image_openCV * self.mask[:, :, np.newaxis]
        elif y < self.segments_slic.shape[0]:
            x = x - self.image_TK.width() - self.padding_between_images
            y = y - self.image_TK.height()

            clicked_label = self.segments_slic[y][x]
            self.result_image_openCV[self.segments_slic == clicked_label] = 0
            self.mask[self.segments_slic == clicked_label] = 0

        result_image_PIL = Image.fromarray(self.result_image_openCV)
        self.result_image_TK = ImageTk.PhotoImage(result_image_PIL)
        self.canvas.create_image(self.result_image_TK.width() + self.padding_between_images, 0, anchor=NW,
                                 image=self.result_image_TK)

    def _load_image(self):
        self.image_path = filedialog.askopenfilename()

        if len(self.image_path) > 0:
            self.image_openCV = cv2.imread(self.image_path)
            self.image_openCV = cv2.cvtColor(self.image_openCV, cv2.COLOR_BGR2RGB)

            self.image_openCV = self._resize_image(self.image_openCV)

            # calculate slic and mark the superpixels
            self.segments_slic = slic(img_as_float(self.image_openCV), n_segments=500, compactness=20, sigma=1,
                                      start_label=0)
            image_boundaries = mark_boundaries(img_as_float(self.image_openCV), self.segments_slic,
                                               color=(255, 254, 253))

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

    # improve the resize with better methods
    def _resize_image(self, image):
        if image.shape[0] >= 1600 or image.shape[1] >= 1600:
            width = image.shape[1] // 3
            height = image.shape[0] // 3
        elif image.shape[0] >= 800 or image.shape[1] >= 800:
            width = image.shape[1] // 2
            height = image.shape[0] // 2
        else:
            width = image.shape[1]
            height = image.shape[0]

        return cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)

    def _save_result(self):
        save_directory_path = filedialog.askdirectory()
        if len(save_directory_path) > 0:
            file_name = self.image_path.split('/')
            file_name = file_name[len(file_name) - 1]

            self.result_image_openCV = cv2.cvtColor(self.result_image_openCV, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f"{save_directory_path}/result_{file_name}", self.result_image_openCV)
            self.result_image_openCV = cv2.cvtColor(self.result_image_openCV, cv2.COLOR_BGR2RGB)


if __name__ == "__main__":
    root = Tk()

    app = SuperpixelApp(master=root)

    app.mainloop()
