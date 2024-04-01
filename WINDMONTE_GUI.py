import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pickle

class LoadSection:
    def __init__(self, master, title):
        self.master = master
        self.title = title

        self.frame = tk.Frame(master, borderwidth=2, relief="groove")
        self.frame.pack(side=tk.LEFT, padx=10, pady=10)

        # Create a title label for the section
        self.title_label = tk.Label(self.frame, text=self.title, font=("Helvetica", 14, "bold"))
        self.title_label.pack(side=tk.TOP, pady=5)

        # Create a button to select a data file
        self.load_button = tk.Button(self.frame, text="Select Data File", command=self.load_data_file)
        self.load_button.pack(side=tk.TOP, pady=5)

        self.button = tk.Button(self.frame, text="Display VOI Keys", command=self.display_keys)
        self.button.pack(side=tk.TOP, pady=5)

        # Create green lights for variables x, y, and z
        self.x_light = tk.Label(self.frame, text="Run Data", bg="red", width=12)
        self.x_light.pack(side=tk.LEFT, padx=5)
        self.y_light = tk.Label(self.frame, text="U_Systematic", bg="red", width=12)
        self.y_light.pack(side=tk.LEFT, padx=5)
        self.z_light = tk.Label(self.frame, text="U_Random", bg="red", width=12)
        self.z_light.pack(side=tk.LEFT, padx=5)

        

    def load_data_file(self):
        global RunData, U_systematic, U_random
        # Open a file dialog to select a data file
        file_path = filedialog.askopenfilename(title="Select Data File", filetypes=[("Pickle files", "*.pkl")])
        if file_path:
            print(f"Selected file: {file_path}")
            # Implement data loading logic 
            try:
                with open(file_path, 'rb') as f:
                    RunData,U_systematic,U_random = pickle.load(f)
            except Exception as e:
                print(f"Error loading data: {e}")

            # Check if variables x, y, and z are present and update lights accordingly
            if 'RunData' in globals() or 'RunData' in locals():
                self.x_light.config(bg="green")
            if 'U_systematic' in globals() or 'U_systematic' in locals():
                self.y_light.config(bg="green")
            if 'U_random' in globals() or 'U_random' in locals():
                self.z_light.config(bg="green")

    def display_keys(self):
        # Get the keys from the dictionary
        keys = list(RunData[1].keys())

        # Create a new window to display the keys
        self.keys_window = tk.Toplevel(self.master)
        self.keys_window.title("VOI Keys")

        # Create a label to display the keys
        label = tk.Label(self.keys_window, text='\n'.join(keys))
        label.pack()

class Run_Section:
    def __init__(self, master, title):
        self.master = master
        self.title = title

        self.frame = tk.Frame(master, borderwidth=2, relief="groove")
        self.frame.pack(side=tk.LEFT, padx=10, pady=10)

        # Create a title label for the section
        self.title_label = tk.Label(self.frame, text=self.title, font=("Helvetica", 14, "bold"))
        self.title_label.pack(side=tk.TOP, pady=5)

        # Create a label and entry for the x variable
        self.x_variable_label = tk.Label(self.frame, text="X Variable:")
        self.x_variable_label.pack(side=tk.TOP, padx=5)
        self.x_variable = tk.Entry(self.frame)
        self.x_variable.pack(side=tk.TOP, padx=5)
        self.set_default_entry(self.x_variable, 'AlphaC')

        # Create a label and text widget for the y variables
        self.y_variables_label = tk.Label(self.frame, text="Y Variables (Separate by comma):")
        self.y_variables_label.pack(side=tk.TOP, padx=5)
        self.y_variables = tk.Text(self.frame, height=1, width=20)
        self.y_variables.pack(side=tk.TOP, padx=5)

        # Create buttons for specific behaviors
        self.errorbars_button = tk.Button(self.frame, text="Plot with error bars", command=self.plot_errorbars)
        self.errorbars_button.pack(side=tk.TOP, pady=5)
        self.boxplot_button = tk.Button(self.frame, text="Boxplot", command=self.boxplot)
        self.boxplot_button.pack(side=tk.TOP, pady=5)
        self.u_voi_button = tk.Button(self.frame, text="Plot Uncertainty Magnitudes", command=self.plot_U_VOI)
        self.u_voi_button.pack(side=tk.TOP, pady=5)
        self.u_and_UPC_button = tk.Button(self.frame, text="Uncertainty Magnitudes & UPCs", command=self.plot_U_and_UPCs)
        self.u_and_UPC_button.pack(side=tk.TOP, pady=5)
        self.upcs_button = tk.Button(self.frame, text="Plot UPCs", command=self.plot_UPCs)
        self.upcs_button.pack(side=tk.TOP, pady=5)

    def set_default_entry(self, entry, default_text):
        entry.insert('0',default_text)


    def plot_errorbars(self):
        # Implement plot_errorbars method
        print(f"Plotting error bars for {self.title}")
        xvar = self.x_variable.get()
        yvars = self.y_variables.get("1.0", "end-1c").split(',')
        RunData.plot_errorbars(xvar,yvars,ncols=min([3,len(yvars)]))
        plt.show()

    def plot_U_VOI(self):
        # Implement plot_U_VOI method
        print(f"Plotting U_VOI for {self.title}")
        xvar = self.x_variable.get()
        yvars = self.y_variables.get("1.0", "end-1c").split(',')
        RunData.plot_U_VOI(xvar,yvars,ncols=min([3,len(yvars)]))
        plt.show()

    def boxplot(self):
        # Implement plot_U_VOI method
        print(f"Creating boxplot for {self.title}")
        xvar = self.x_variable.get()
        yvars = self.y_variables.get("1.0", "end-1c").split(',')
        RunData.boxplot(xvar,yvars,ncols=min([3,len(yvars)]))
        plt.show()

    def plot_UPCs(self):
        # Implement plot_UPCs method
        print(f"Plotting UPCs for {self.title}")
        xvar = self.x_variable.get()
        yvars = self.y_variables.get("1.0", "end-1c").split(',')
        RunData.plot_UPCs(xvar,yvars,ncols=min([3,len(yvars)]))
        plt.show()

    def plot_U_and_UPCs(self):
        # Implement plot_UPCs method
        print(f"Plotting UPCs for {self.title}")
        xvar = self.x_variable.get()
        yvars = self.y_variables.get("1.0", "end-1c").split(',')
        RunData.plot_U_and_UPCs(xvar,yvars,ncols=min([3,len(yvars)]))
        plt.show()

class DataPoint_Section:
    def __init__(self, master, title):
        self.master = master
        self.title = title

        self.frame = tk.Frame(master, borderwidth=2, relief="groove")
        self.frame.pack(side=tk.LEFT, padx=10, pady=10)

        # Create a title label for the section
        self.title_label = tk.Label(self.frame, text=self.title, font=("Helvetica", 14, "bold"))
        self.title_label.pack(side=tk.TOP, pady=5)

        # Create a label and entry for the x variable
        self.x_variable_label = tk.Label(self.frame, text="X Variable:")
        self.x_variable_label.pack(side=tk.TOP, padx=5)
        self.x_variable = tk.Entry(self.frame)
        self.x_variable.pack(side=tk.TOP, padx=5)

        # Create a label and text widget for the y variables
        self.y_variables_label = tk.Label(self.frame, text="Y Variables:")
        self.y_variables_label.pack(side=tk.TOP, padx=5)
        self.y_variables = tk.Text(self.frame, height=4, width=30)
        self.y_variables.pack(side=tk.TOP, padx=5)


        # Create buttons for specific behaviors
        self.errorbars_button = tk.Button(self.frame, text="Plot Error Bars", command=self.plot_errorbars)
        self.errorbars_button.pack(side=tk.TOP, pady=5)
        self.u_voi_button = tk.Button(self.frame, text="Plot U_VOI", command=self.plot_U_VOI)
        self.u_voi_button.pack(side=tk.TOP, pady=5)
        self.upcs_button = tk.Button(self.frame, text="Plot UPCs", command=self.plot_UPCs)
        self.upcs_button.pack(side=tk.TOP, pady=5)

    def set_default_entry(self, entry, default_text):
        entry.insert('0',default_text)


    def plot_errorbars(self):
        # Implement plot_errorbars method
        print(f"Plotting error bars for {self.title}")

    def plot_U_VOI(self):
        # Implement plot_U_VOI method
        print(f"Plotting U_VOI for {self.title}")

    def plot_UPCs(self):
        # Implement plot_UPCs method
        print(f"Plotting UPCs for {self.title}")

class VOI_Section:
    def __init__(self, master, title):
        self.master = master
        self.title = title

        self.frame = tk.Frame(master, borderwidth=2, relief="groove")
        self.frame.pack(side=tk.LEFT, padx=10, pady=10)

        # Create a title label for the section
        self.title_label = tk.Label(self.frame, text=self.title, font=("Helvetica", 14, "bold"))
        self.title_label.pack(side=tk.TOP, pady=5)

        # Create a label and entry for the VOI variable
        self.VOI_variable_label = tk.Label(self.frame, text="VOI:")
        self.VOI_variable_label.pack(side=tk.TOP, padx=5)
        self.VOI_variable = tk.Entry(self.frame)
        self.VOI_variable.pack(side=tk.TOP, padx=5)

        # Create a label and text widget for the data point
        self.point_label = tk.Label(self.frame, text="Point #:")
        self.point_label.pack(side=tk.TOP, padx=5)
        self.point = tk.Entry(self.frame)
        self.point.pack(side=tk.TOP, padx=5)

        # Create buttons for specific behaviors
        self.histogram_button = tk.Button(self.frame, text="Plot Histogram", command=self.plot_histogram)
        self.histogram_button.pack(side=tk.TOP, pady=5)

        self.QQplot_button = tk.Button(self.frame, text="QQ Plot: Normal Distribution Test", command=self.QQplot)
        self.QQplot_button.pack(side=tk.TOP, pady=5)
        
        self.upcs_button = tk.Button(self.frame, text="Plot UPCs", command=self.plot_UPCs)
        self.upcs_button.pack(side=tk.TOP, pady=5)

    def plot_histogram(self):
        # Implement plot_histogram method
        print(f"Plotting histogram for {self.title}")
        VOI = RunData[int(self.point.get())][self.VOI_variable.get()]
        VOI.plot_histogram()

    def QQplot(self):
        # Implement plot Quantile-Quantile method
        print(f"Plotting Quantile-Quantile for {self.title}")
        VOI = RunData[int(self.point.get())][self.VOI_variable.get()]
        VOI.plot_QQ()

    def plot_UPCs(self):
        # Implement plot_UPCs method
        print(f"Plotting UPCs for {self.title}")
        VOI = RunData[int(self.point.get())][self.VOI_variable.get()]
        VOI.plot_UPCs()


# Create the main application window
root = tk.Tk()
root.title("WINDMONTE GUI")

# Create a section for loading data
load_data_section = LoadSection(root, "Load Data")

# Create sections for Run, DataPoint, and VOI
run_section = Run_Section(root, "RunData")
#datapoint_section = DataPoint_Section(root, "DataPoint")  # For later development
voi_section = VOI_Section(root, "VOI")

# Perform load of default file
global RunData, U_systematic, U_random
# Open a file dialog to select a data file
file_path = "WINDMONTE_outputs.pkl"
print('Loading default file {}'.format(file_path))
try:
    with open(file_path, 'rb') as f:
        RunData,U_systematic,U_random = pickle.load(f)
except Exception as e:
    print(f"Error loading default data file {file_path}: {e}")

""" # Check if variables x, y, and z are present and update lights accordingly
if 'RunData' in globals() or 'RunData' in locals():
    LoadSection.x_light.config(bg="green")
if 'U_systematic' in globals() or 'U_systematic' in locals():
    LoadSection.y_light.config(bg="green")
if 'U_random' in globals() or 'U_random' in locals():
    LoadSection.z_light.config(bg="green") """

# Start the Tkinter event loop
root.mainloop()

















