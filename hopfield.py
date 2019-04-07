# import tkinter.font
import numpy as np
import matplotlib.pyplot as plt
import io
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import tkinter as tk
from tkinter import ttk

# LARGE_FONT = ("Verdana", 12)

# Substitute black (=1) => -1, white (=0) =>0
""" it is a bit confusing, as in txt file black equals 1 and white equals 0,
 in hopfield form black equals -1 and white equal 1"""
def substituteMatrices(s):
    for matrix in range(len(s)):
        for i in range(len(s[matrix])):
            for j in range(len(s[matrix][i])):
                s[matrix][i][j] = -1 if s[matrix][i][j] == 1 else 1
    return s

def matrix_to_string(m, dim1, dim2):
    s = ""
    for i in range(dim1):
        for j in range(dim2):
            s = s + ('1 ' if m[dim1*i+j] < 0 else '0 ')
        s = s + chr(10)
    return s

def clearTxt():
    open('output.txt', 'w').close()

def writeToTXT(S, dim1, dim2):
    for i in range(len(S)):
        mat = matrix_to_string(S[i], dim1, dim2)
        with io.open("output.txt", 'a', encoding='utf8') as f:
             f.write(mat + chr(10))


class HopfieldNetwork:
    # Initialize a Hopfield network with N neurons
    def __init__(self, N):
        self.N = N
        self.W = np.zeros((N, N))
        self.s = np.zeros((N, 1))

    # Apply the Hebbian learning rule. The argument is a matrix S, which contains one sample state per row
    def train(self, S):
        self.W = np.matmul(S.transpose(), S)

    # Run one simulation step
    def runStep(self):
        i = np.random.randint(0, self.N)    # Random bit
        a = np.matmul(self.W[i, :], self.s) # Activation function
        if a < 0:
            if self.s[i] == 1:
                self.s[i] = -1
                self.changeBuffer.append(i) #add record of changed bits to the outpout
        else:
            if self.s[i] == -1:
                self.s[i] = 1
                self.changeBuffer.append(i) #add record of changed bits to the outpout

    # Starting with a given state, execute the update rule N times and return the resulting state
    def run(self, state, steps):
        self.s = state.copy()
        self.changeBuffer = []
        for i in range(steps):
            self.runStep()
        return self.changeBuffer    # list of changed bits

#######################################################################################################################

def tokenizer(fname):
    with open(fname) as f:
        chunk = []
        for line in f:
            if 'HEAD'in line:
                continue
            if 'END' in line:
                yield chunk
                chunk = []
                continue
            chunk.append(line)


class SeaofBTCapp(tk.Tk):

    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        # tk.Tk.iconbitmap(self, default="clienticon.ico")
        tk.Tk.wm_title(self, "Hopfield's network")
        tk.Tk.resizable(self, width=False, height=False)

        self.frames = {}
        frame = StartPage(self)
        self.frames[StartPage] = frame
        frame.grid(row=0, column=0, sticky="nsew")
        self.show_frame(StartPage)

    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()

class StartPage(tk.Frame):

    def loadandlearn(self):
        # Load matrices from txt
        self.arrays = [np.loadtxt(A) for A in tokenizer('input.txt')]
        self.arraysReshaped = [np.reshape(a, (1, -1)) for a in self.arrays]

        # Find dimenses of matrices
        self.dim1 = len(self.arrays[0])
        self.dim2 = len(self.arrays[0][0])

        # Create vector of matrices
        self.arraysReshapedSubstitued = substituteMatrices(self.arraysReshaped)
        self.S = np.concatenate(self.arraysReshapedSubstitued)  # Main matrix of all samples

        # Get number of pictures
        self.memories = len(self.S)

        # Declaration of plots
        self.plotsLearned = self.declarePlots(0)
        self.plotsDamaged = self.declarePlots(1)
        self.plotsReconstructed = self.declarePlots(2)

        # Update canvas
        self.canvas.draw()

        # Init network
        self.HN = HopfieldNetwork(self.dim1 * self.dim2 )

        # Train Hopfield network
        self.HN.train(self.S)

        # Draw to canvas
        for pic in range(self.memories):
            state = (self.S[pic, :].reshape(self.dim1 * self.dim2, 1)).copy()
            self.plotsLearned[pic].imshow(state.reshape(self.dim1, self.dim2), "binary_r")
        self.canvas.draw()

    def declarePlots(self, column):
        plots = [None] * self.memories
        for i in range(self.memories):
            plots[i] = self.f.add_subplot(self.memories, 3, column + 1 + i*3)
            plots[i].set_xticks([], [])
            plots[i].set_yticks([], [])
            if i == 0:
                if column == 0:
                    plots[i].set_title("Learned patterns")
                elif column == 1:
                    plots[i].set_title("Damaged patterns")
                elif column == 2:
                    plots[i].set_title("Reconstruction")
        return plots

    def learn_parameters(self):
        self.errors = int(self.entry_errors.get())
        self.iterations = int(self.entry_iter.get())

    def damage(self):
        self.learn_parameters() # load values from entry windows
        self.damaged = []
        clearTxt()

        # Write txt title
        with io.open("output.txt", 'a', encoding='utf8') as f:
             f.write("Damaged matrices:" + chr(10))

        # Clear animation image
        self.clear_animation()

        for pic in range(self.memories):
            state = (self.S[pic, :].reshape(self.dim1 * self.dim2, 1)).copy()
            bitsChanged = []
            i = 0

            # Flip few bits
            while i < self.errors:
                index = np.random.randint(0, self.dim1 * self.dim2)
                if index not in bitsChanged:
                    state[index][0] = state[index][0] * (-1)
                    bitsChanged.append(index)
                    i += 1

            self.damaged.append(state)

            # Draw to canvas
            self.plotsDamaged[pic].imshow(state.reshape(self.dim1, self.dim2), "binary_r")

        # Write matrix to txt
        writeToTXT(self.damaged, self.dim1, self.dim2)

        # Update canvas
        self.canvas.draw()

    def clear_animation(self):
        for pic in range(self.memories):
            self.plotsReconstructed[pic].clear()
            self.plotsReconstructed[pic].set_xticks([], [])
            self.plotsReconstructed[pic].set_yticks([], [])
            if pic == 0:
                self.plotsReconstructed[pic].set_title("Reconstruction")

    def run(self):
        # Update parameter iterations
        self.learn_parameters()

        # Clear images for animation
        self.clear_animation()

        # Clear repaired matrices for repeated reconstructions
        self.repaired = []
        for pic in range(self.memories):
            self.repaired.append(self.damaged[pic].copy())

        # Write txt title
        with io.open("output.txt", 'a', encoding='utf8') as f:
            f.write("----------------------------------------------------------" + chr(10) + "Reconstructed matrices:" + chr(10))

        # For each picture count Hopfield network
        for pic in range(self.memories):

            # Run HN nad get buffer of changed bits
            self.changeBuffer = self.HN.run(self.repaired[pic], self.iterations)

            # Draw default damaged patterns
            self.plotsReconstructed[pic].imshow(self.repaired[pic].reshape(self.dim1, self.dim2), "binary_r")
            self.canvas.draw()

            # Make animated changes in graph
            for i in range(len(self.changeBuffer)):
                self.repaired[pic][self.changeBuffer[i]] = -1 * self.repaired[pic][self.changeBuffer[i]]
                plt.pause(0.3)
                self.plotsReconstructed[pic].imshow(self.repaired[pic].reshape(self.dim1, self.dim2), "binary_r")
                self.canvas.draw()

        # Write matrix to txt
        writeToTXT(self.repaired, self.dim1, self.dim2)



    def __init__(self, parent):
        tk.Frame.__init__(self, parent)

        # Title
        self.lbl_settings= tk.Label(self, text="Settings", font="Helvetica 12 bold ")

        # Labels
        self.lbl_errors = tk.Label(self, text="Errors:")
        self.lbl_iter = tk.Label(self, text="Iterations:")
        self.lbl_info = tk.Label(self, text="Author: Dominik Jasek @ BUT FME 2019")

        # Entries
        self.entry_errors = tk.Entry(self, width=10)
        self.entry_errors.insert(0, 4)
        self.entry_iter = tk.Entry(self, width=10)
        self.entry_iter.insert(0, 500)

        # Buttons
        self.btn_ll = ttk.Button(self, text="LOAD & LEARN", command=lambda: self.loadandlearn())
        self.btn_damage = ttk.Button(self, text="DAMAGE", command=lambda: self.damage())
        self.btn_run = ttk.Button(self, text="RUN HOPFIELD NETWORK", command=lambda: self.run())

        # Separators
        tk.ttk.Separator(self, orient="horizontal").grid(column=1, row=4, columnspan=4, sticky='ew')

        # Positions
        self.lbl_settings.grid(row=1, column=1, sticky='e')
        self.lbl_errors.grid(row=2, column=1, sticky='e')
        self.lbl_iter.grid(row=3, column=1, sticky='e')
        self.entry_errors.grid(row=2, column=2, sticky='w')
        self.entry_iter.grid(row=3, column=2, sticky='w')
        self.lbl_info.grid(row=5, column=1, columnspan = 3)
        self.btn_ll.grid(row=1, column=3, sticky='wens')
        self.btn_damage.grid(row=2, column=3, sticky='wens')
        self.btn_run.grid(row=3, column=3, sticky='wens')

        # Create figure
        self.f = Figure(figsize=(6, 6), dpi=100)

        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.f, self)
        self.canvas._tkcanvas.grid(row=6, column=1, columnspan=3, sticky='nesw')
        self.canvas.draw()



# Main programm
app = SeaofBTCapp()
app.mainloop()