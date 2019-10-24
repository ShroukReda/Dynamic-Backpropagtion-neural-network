import cv2
import matplotlib.pyplot as plt
import tkinter as tk
import glob
from tkinter import ttk
from PIL import ImageTk, Image
import Task1Tain as T1
from tkinter import *
def on_select(event=None):
    print('----------------------------')
    bias.bind('<<ComboboxSelected>>', on_select)
    AF.bind('<<ComboboxSelected>>', on_select)
    MyFeatures,MyClasses=T1.loading()
    Classes, NumOfHidd, NumofNeurons, Bias, Func,D= T1.Train(MyFeatures,MyClasses,Ep.get(),R.get(),R2.get(),ListOfNumOfNeurons,comboboxes[0].get(),comboboxes[1].get())
    A, M = T1.Test(Classes, NumOfHidd, NumofNeurons, Bias, Func,MyFeatures,D)
    print(A)
    print(M)
    '''
    f1,f2,f3,f4,c1,c2 = T1.SetClassesAndFeatures(comboboxes[0].get(),comboboxes[1].get(),comboboxes[2].get(),comboboxes[3].get()) #de l event bta3 zrar OK endhy hna okay foll
    w=T1.MSETrain(f1,f2,f3,f4,c1,c2,R2.get(),R.get(),comboboxes[4].get())
    Acc,M=T1.Test(f1,f2,f3,f4,c1,c2,w)
    T1.Draw(f1,f2,f3,f4,w)
    print(Acc)
    print(M)
    '''
ListOfNumOfNeurons=[]
def Clear(event=None):
    ListOfNumOfNeurons.append(R22.get())
    R22.delete(0,END)
    R22.pack()
top = tk.Tk()
top.title("Task1")
top.geometry("712x400")
top.configure(background="black")

comboboxes = []
bias = ttk.Combobox(top, values=("With Bias", "Without Bias"), state='readonly')
bias.set("Bias")
bias.pack()
comboboxes.append(bias)

AF = ttk.Combobox(top, values=("Sigmoid", "Hyperbolic Tangent"), state='readonly')
AF.set("Activation Function")
AF.pack()
comboboxes.append(AF)

var2 = StringVar()
label2 = Label( top, textvariable=var2, relief=RAISED )
var2.set("Enter # of hidden layers")
label2.pack()
R2=tk.Entry(top,bd=5)
R2.pack()

var3 = StringVar()
labe22 = Label( top, textvariable=var3, relief=RAISED )
var3.set("Enter # of neurons")
labe22.pack()
R22=Entry(top,bd=5)
R22.pack()

NextLayer = tk.Button(top, text="NextLayer !", command=Clear)
NextLayer.pack()

var = StringVar()
label = Label( top, textvariable=var, relief=RAISED )
var.set("Enter Learning Rate")
label.pack()
R=tk.Entry(top,bd=5)
R.pack()

var1 = StringVar()
label1 = Label( top, textvariable=var1, relief=RAISED )
var1.set("Enter # of Epochs")
label1.pack()
Ep=tk.Entry(top,bd=5)
Ep.pack()


TrainAndTest = tk.Button(top, text="TrainAndTest !", command=on_select)
TrainAndTest.pack()
top.mainloop()

