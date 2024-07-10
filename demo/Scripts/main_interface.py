import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import interfaceIDENTIFICATION
import interfaceVERIFICATION
import os

class MainApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("CHOOSE THE APPLICATION")
        self.geometry("600x400")

        background_image_path = os.path.join(os.path.dirname(__file__), 'pexels-rudy-kirchner-278171-831084.jpg')
        background_image = Image.open(background_image_path)
        background_photo = ImageTk.PhotoImage(background_image)

        self.canvas = tk.Canvas(self, width=600, height=400)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.create_image(0, 0, image=background_photo, anchor="nw")

        self.button1 = tk.Button(self, text=" WHALE VERIFICATION ", command=self.open_verification)
        self.button1_window = self.canvas.create_window(125, 200, anchor="nw", window=self.button1)

        self.button2 = tk.Button(self, text=" WHALE IDENTIFICATION ", command=self.open_identification)
        self.button2_window = self.canvas.create_window(325, 200, anchor="nw", window=self.button2)

        self.background_photo = background_photo

    def open_verification(self):
        self.destroy()
        app = interfaceVERIFICATION.VerificationApp()
        app.mainloop()

    def open_identification(self):
        self.destroy()
        app = interfaceIDENTIFICATION.IdentificationApp()
        app.mainloop()

if __name__ == "__main__":
    app = MainApp()
    app.mainloop()