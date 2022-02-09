from tkinter import *
import mss
import mss.tools
import threading as th 
import tensorflow as tf
import numpy as np
import cv2

class Board(object):

    count= 0
    S
    PEN_SIZE = 40
    PRED = 4
    FILE_NAME = "Temp"
    FLAG = False
    OUTPUT = Image
    MODEL = tf.keras.models.load_model('handToD.model')
    X_DATA = []
    

    def __init__(self):
        self.S = th.Timer(3.0, self.collapse)
        self.frame = Tk()
        self.frame2 = Toplevel()   
        
        self.frame.geometry("500x500+1200+100")
        self.frame2.geometry("100x100+1710+100")
        
        self.c = Canvas(self.frame, bg='black', width=500, height=500)
        self.c.grid(row=1, columnspan=5)

        #self.Texty = StringVar(value='-')
        #print(self.Texty.get())
        global txt 
        txt = StringVar()
        global lb1
        lb1 = Label(self.frame2, textvariable=txt, anchor=NW, justify=LEFT, wraplength=398, font=('Arial',32)).pack(pady=20)
        txt.set("-")
        
        
        self.setup()
        self.frame.mainloop()

    def setup(self):
        self.old_x = None
        self.old_y = None
        self.line_width = self.PEN_SIZE
        self.color = 'white' 
    
        self.c.bind('<B1-Motion>', self.paint)
        self.c.bind('<ButtonRelease-1>', self.reset)

    def paint(self, event):
        
        if self.FLAG == False and not self.S.is_alive():
            self.Flag = True
            self.S.start()
        
        self.line_width = self.PEN_SIZE
        paint_color = 'white'
        if self.old_x and self.old_y:
            self.c.create_line(self.old_x, self.old_y, event.x, event.y,
                               width=self.line_width, fill=paint_color,
                               capstyle=ROUND, smooth=TRUE, splinesteps=36)
        self.old_x = event.x
        self.old_y = event.y
        

    def Pred(self):
        img = cv2.imread('TEMP.png', cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (28, 28))

        import matplotlib.pyplot as plt

        plt.imshow(img)
        plt.show()

        self.X_DATA.append(img)

        logged = np.array(self.X_DATA)
        logged = logged.astype('float32') / 255

        self.X_DATA.clear()

        pred = self.MODEL.predict([logged])
        #print(np.argmax(pred[0]))
        txt.set(np.argmax(pred[0]))
        
        #print()
          

    def save_as_png(self):
        with mss.mss() as sct:
            # The screen part to capture
            monitor = {"top": 140, "left": 1210, "width": 500, "height": 500}
            output = "TEMP.png".format(**monitor)

            # Grab the data
            sct_img = sct.grab(monitor)

            # Save to the picture file
            mss.tools.to_png(sct_img.rgb, sct_img.size, output=output)
            self.OUTPUT = output
            #print(output)
        
       
    def collapse(self):
        #print("saving..")
        self.save_as_png()
        #print("taking..")
        self.c.delete("all")
        self.S.cancel()
        self.Flag = False
        self.Pred()
        self.S = th.Timer(3.0, self.collapse)

    def reset(self, event):
        self.old_x, self.old_y = None, None



if __name__ == '__main__':
    Board()
    