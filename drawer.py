""""
Very ugly code just to capture drawn digits, modyfing paint program
created by Dave Mitchell.

GUI done Tkinter (only because Qt was refusing to cooperate ;) )

"""
from Tkinter import *


b1 = "up"
xold, yold = None, None
segments = 0
drawing_area = None
def draw_main():
    global drawing_area
    root = Tk()
    root.wm_title("SeMPowisko2014")
    drawing_area = Canvas(root, width=280, height=280, background='white')
    drawing_area.pack()
    drawing_area.bind("<Motion>", motion)
    drawing_area.bind("<ButtonPress-1>", b1down)
    drawing_area.bind("<ButtonRelease-1>", b1up)

    drawing_area.bind("<Button-3>", b2down)

    root.mainloop()

def b2down(event):
    global segments, drawing_area
    print "EVENT"
    event.widget.delete('all')


def b1down(event):
    global b1
    b1 = "down"           # you only want to draw when the button is down
                          # because "Motion" events happen -all the time-


def b1up(event):
    global b1, xold, yold
    b1 = "up"
    xold = None           # reset the line when you let go of the button
    yold = None

def motion(event):
    global segments, drawing_area
    if b1 == "down":
        global xold, yold
        if xold is not None and yold is not None:
            segments += 0
            if segments % 100 == 0:
                drawing_area.update()
                drawing_area.postscript(file="digit.eps")

            event.widget.create_line(xold,yold,event.x,event.y,smooth=TRUE, width=7)
                          # here's where you draw it. smooth. neat.
        xold = event.x
        yold = event.y
import threading

if __name__ == "__main__":
    threading.Thread(target=draw_main).run()

