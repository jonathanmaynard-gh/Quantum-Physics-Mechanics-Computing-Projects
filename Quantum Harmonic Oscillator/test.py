import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import tkinter as tk
from tkinter import ttk
from scipy.special import hermite, factorial
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

plt.style.use('dark_background')

hbar = 1.0  #reduced Planck constant

colors = [(0, 0, 0.5), (0, 0.5, 1), (0, 0.8, 0.8), (0.9, 1, 0), (1, 0.5, 0)]
cmap_quantum = LinearSegmentedColormap.from_list("quantum", colors, N=256)

try:
    mpl.colormaps.register(cmap=cmap_quantum, name="quantum")
except AttributeError:
    try:
        plt.cm.register_cmap(name="quantum", cmap=cmap_quantum)
    except:
        print("Could not register custom colormap, using 'viridis' instead")

#calc the normalized wavefunction for the 2D quantum harmonic oscillator
def wavefunction(nx, ny, x, y, omega, mass):
    """
    Calculate the normalized wavefunction for a 2D quantum harmonic oscillator.
    
    Parameters:
    -----------
    nx, ny : int
        Quantum numbers for x and y directions
    x, y : array_like
        Position coordinates
    omega : float
        Angular frequency of the oscillator
    mass : float
        Mass of the particle
    
    Returns:
    --------
    array_like
        The normalized wavefunction
    """
    #alpha parameter (related to characteristic length)
    alpha = np.sqrt(mass * omega / hbar)
    
    #normalization factors
    norm_x = 1.0 / np.sqrt(2**nx * factorial(nx)) * (alpha / np.pi)**0.25
    norm_y = 1.0 / np.sqrt(2**ny * factorial(ny)) * (alpha / np.pi)**0.25
    
    #clculate the Hermite polynomials using scipy's implementation
    hermite_x = hermite(nx)(alpha * x)
    hermite_y = hermite(ny)(alpha * y)
    
    #exponential factor
    exp_factor = np.exp(-0.5 * alpha**2 * (x**2 + y**2))
    
    #normalized wavefunction
    return norm_x * norm_y * exp_factor * hermite_x * hermite_y

class SimpleSlider(tk.Frame):
    def __init__(self, parent, min_val, max_val, resolution=1, default=0, label="", variable=None, width=150):
        super().__init__(parent, bg="#444444")
        
        self.min_val = min_val
        self.max_val = max_val
        self.resolution = resolution
        
        if variable is None:
            if isinstance(min_val, int) and isinstance(max_val, int) and resolution == 1:
                self.variable = tk.IntVar(value=default)
            else:
                self.variable = tk.DoubleVar(value=default)
        else:
            self.variable = variable
            self.variable.set(default)
        
        if label:
            self.label = tk.Label(self, text=label, font=('Arial', 10, 'bold'), 
                                 bg="#444444", fg="white", width=5, anchor='e')
            self.label.pack(side=tk.LEFT, padx=5)
        
        self.dec_button = tk.Button(self, text="-", command=self.decrement,
                                  bg="gray", fg="white", width=2, 
                                  font=('Arial', 10, 'bold'))
        self.dec_button.pack(side=tk.LEFT, padx=2)
        
        self.spinbox = tk.Spinbox(self, from_=min_val, to=max_val, increment=resolution,
                               textvariable=self.variable, width=5, 
                               bg="white", fg="black", font=('Arial', 10))
        self.spinbox.pack(side=tk.LEFT, padx=2)
        
        self.inc_button = tk.Button(self, text="+", command=self.increment,
                                  bg="gray", fg="white", width=2,
                                  font=('Arial', 10, 'bold'))
        self.inc_button.pack(side=tk.LEFT, padx=2)
        
        self.value_display = tk.Label(self, textvariable=self.variable, width=4,
                                   bg="#444444", fg="yellow", font=('Arial', 10, 'bold'))
        self.value_display.pack(side=tk.LEFT, padx=5)

        self.variable.trace_add("write", self.validate)
    
    def decrement(self):
        current = self.variable.get()
        new_val = max(self.min_val, current - self.resolution)
        self.variable.set(round(new_val, 2) if self.resolution < 1 else int(new_val))
    
    def increment(self):
        current = self.variable.get()
        new_val = min(self.max_val, current + self.resolution)
        self.variable.set(round(new_val, 2) if self.resolution < 1 else int(new_val))
    
    def validate(self, *args):
        try:
            current = self.variable.get()
            if current < self.min_val:
                self.variable.set(self.min_val)
            elif current > self.max_val:
                self.variable.set(self.max_val)
        except:
            self.variable.set(self.min_val)

def set_modern_style():
    style = ttk.Style()
    
    if 'clam' in style.theme_names():
        style.theme_use('clam')
    
    style.configure('TButton', 
                    font=('Arial', 10, 'bold'),
                    padding=5)

    style.configure('TLabel',
                   font=('Arial', 10))

    style.configure('TFrame')
    
    style.configure('TScale')

class QuantumHarmonicOscillatorApp:
    def __init__(self, master):
        self.master = master
        master.title("Quantum Harmonic Oscillator Visualizer")
        master.geometry("1200x800")
        master.configure(bg='#333333')
        
        self.contour_plot = None

        self.current_params = {'nx': 0, 'ny': 0, 'omega': 1.0, 'mass': 1.0}
        
        set_modern_style()

        main_container = tk.Frame(master, bg='#333333')
        main_container.pack(fill=tk.BOTH, expand=True)

        self.control_panel = tk.Frame(main_container, bg='#444444', width=300)
        self.control_panel.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        self.control_panel.pack_propagate(False)  # Keep fixed width

        viz_container = tk.Frame(main_container, bg='#333333')
        viz_container.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.plot_frame = tk.Frame(viz_container, bg='#222222')
        self.plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.info_frame = tk.Frame(viz_container, bg='#444444', height=40)
        self.info_frame.pack(side=tk.BOTTOM, fill=tk.X)

        self.nx_var = tk.IntVar(value=0)
        self.ny_var = tk.IntVar(value=0)
        self.omega_var = tk.DoubleVar(value=1.0)
        self.mass_var = tk.DoubleVar(value=1.0)
        self.energy_var = tk.StringVar()
        self.state_var = tk.StringVar()

        self.create_controls()

        self.setup_plot()

        self.create_info_display()

        self.update_plot()
    
    def create_info_display(self):
        """Create the information display panel"""
        self.energy_label = tk.Label(self.info_frame, textvariable=self.energy_var, 
                               font=('Arial', 12, 'bold'), bg='#444444', fg='white')
        self.energy_label.pack(side=tk.LEFT, padx=10, pady=5)
        
        self.state_label = tk.Label(self.info_frame, textvariable=self.state_var, 
                               font=('Arial', 12), bg='#444444', fg='white')
        self.state_label.pack(side=tk.RIGHT, padx=10, pady=5)
    
    def setup_plot(self):
        """Setup the matplotlib figure and axis"""
        self.fig = plt.Figure(figsize=(9, 7), dpi=100, facecolor='#333333')
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor('#222222')

        self.canvas = FigureCanvasTkAgg(self.fig, self.plot_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)
        
        toolbar_frame = tk.Frame(self.plot_frame, bg='#333333')
        toolbar_frame.pack(side=tk.BOTTOM, fill=tk.X)
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        
        x = np.linspace(-5.0, 5.0, 200)
        y = np.linspace(-5.0, 5.0, 200)
        X, Y = np.meshgrid(x, y)
        self.X, self.Y = X, Y
        
        potential = 0.5 * (X**2 + Y**2)  #simple harmonic potential V = 0.5*k*r^2, k=1
        self.contour_plot = self.ax.contour(X, Y, potential, 
                                       levels=np.linspace(0, 10, 10), 
                                       colors='white', alpha=0.3, linewidths=0.5)

        try:
            cmap = "quantum"
            self.im = self.ax.imshow(np.zeros((200, 200)), 
                                  extent=(-5, 5, -5, 5),
                                  cmap=cmap, origin="lower", aspect='equal')
        except:
            cmap = "viridis"
            self.im = self.ax.imshow(np.zeros((200, 200)), 
                                  extent=(-5, 5, -5, 5),
                                  cmap=cmap, origin="lower", aspect='equal')
        
        self.colorbar = self.fig.colorbar(self.im)
        self.colorbar.set_label('Probability Density', color='white')
        self.colorbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(plt.getp(self.colorbar.ax.axes, 'yticklabels'), color='white')

        self.ax.set_xlabel("Position x", color='white', fontsize=12)
        self.ax.set_ylabel("Position y", color='white', fontsize=12)
        self.ax.tick_params(colors='white')
        self.title = self.ax.set_title("Quantum Harmonic Oscillator", color='white', fontsize=14)
        
        self.ax.grid(True, linestyle='--', alpha=0.3)
        
        self.ax.axhline(y=0, color='white', linestyle='-', alpha=0.2)
        self.ax.axvline(x=0, color='white', linestyle='-', alpha=0.2)
        
        self.fig.tight_layout()
    
    def create_controls(self):
        """Create UI controls in a vertical layout on the left side"""
        title_label = tk.Label(self.control_panel, text="QUANTUM\nHARMONIC OSCILLATOR", 
                           font=('Arial', 14, 'bold'), 
                           bg='#444444', fg='white')
        title_label.pack(side=tk.TOP, fill=tk.X, pady=15)
        
        quantum_title = tk.Label(self.control_panel, text="QUANTUM NUMBERS", 
                             font=('Arial', 12, 'bold'), 
                             bg='#444444', fg='white')
        quantum_title.pack(side=tk.TOP, fill=tk.X, pady=(20, 5))
        
        self.nx_slider = SimpleSlider(self.control_panel, min_val=0, max_val=5, resolution=1, 
                                  default=0, label="nx:", variable=self.nx_var)
        self.nx_slider.pack(side=tk.TOP, fill=tk.X, padx=20, pady=5)
        
        self.ny_slider = SimpleSlider(self.control_panel, min_val=0, max_val=5, resolution=1, 
                                  default=0, label="ny:", variable=self.ny_var)
        self.ny_slider.pack(side=tk.TOP, fill=tk.X, padx=20, pady=5)
        
        separator1 = tk.Frame(self.control_panel, height=2, bg='#555555')
        separator1.pack(side=tk.TOP, fill=tk.X, padx=20, pady=15)
        
        physical_title = tk.Label(self.control_panel, text="PHYSICAL PARAMETERS", 
                              font=('Arial', 12, 'bold'), 
                              bg='#444444', fg='white')
        physical_title.pack(side=tk.TOP, fill=tk.X, pady=5)
        
        self.omega_slider = SimpleSlider(self.control_panel, min_val=0.1, max_val=5.0, resolution=0.1, 
                                     default=1.0, label="ω:", variable=self.omega_var)
        self.omega_slider.pack(side=tk.TOP, fill=tk.X, padx=20, pady=5)
        
        self.mass_slider = SimpleSlider(self.control_panel, min_val=0.1, max_val=5.0, resolution=0.1, 
                                    default=1.0, label="mass:", variable=self.mass_var)
        self.mass_slider.pack(side=tk.TOP, fill=tk.X, padx=20, pady=5)

        separator2 = tk.Frame(self.control_panel, height=2, bg='#555555')
        separator2.pack(side=tk.TOP, fill=tk.X, padx=20, pady=15)
        
        controls_title = tk.Label(self.control_panel, text="CONTROLS", 
                              font=('Arial', 12, 'bold'), 
                              bg='#444444', fg='white')
        controls_title.pack(side=tk.TOP, fill=tk.X, pady=5)
        
        update_button = tk.Button(self.control_panel, text="UPDATE", command=self.update_plot,
                              font=('Arial', 11, 'bold'), bg='green', fg='white',
                              padx=10, pady=5)
        update_button.pack(side=tk.TOP, padx=20, pady=10)

        separator3 = tk.Frame(self.control_panel, height=2, bg='#555555')
        separator3.pack(side=tk.TOP, fill=tk.X, padx=20, pady=15)

        presets_title = tk.Label(self.control_panel, text="PRESETS", 
                             font=('Arial', 12, 'bold'), 
                             bg='#444444', fg='white')
        presets_title.pack(side=tk.TOP, fill=tk.X, pady=5)

        ground_button = tk.Button(self.control_panel, text="Ground State (0,0)", 
                              command=lambda: self.load_preset(0, 0),
                              font=('Arial', 10), bg='blue', fg='white',
                              width=20)
        ground_button.pack(side=tk.TOP, pady=5)
        
        excited_button = tk.Button(self.control_panel, text="1st Excited (1,0)", 
                               command=lambda: self.load_preset(1, 0),
                               font=('Arial', 10), bg='blue', fg='white',
                               width=20)
        excited_button.pack(side=tk.TOP, pady=5)
        
        symmetric_button = tk.Button(self.control_panel, text="Symmetric (1,1)", 
                                 command=lambda: self.load_preset(1, 1),
                                 font=('Arial', 10), bg='blue', fg='white',
                                 width=20)
        symmetric_button.pack(side=tk.TOP, pady=5)
        
        higher_button = tk.Button(self.control_panel, text="Higher State (3,2)", 
                              command=lambda: self.load_preset(3, 2),
                              font=('Arial', 10), bg='blue', fg='white',
                              width=20)
        higher_button.pack(side=tk.TOP, pady=5)
    
    def load_preset(self, nx, ny):
        """Load a preset quantum state"""
        self.nx_var.set(nx)
        self.ny_var.set(ny)
        self.omega_var.set(1.0)
        self.mass_var.set(1.0)
        self.update_plot()
    
    def update_plot(self):
        """Update the plot data without recreating the plot"""
        nx = self.nx_var.get()
        ny = self.ny_var.get()
        omega = self.omega_var.get()
        mass = self.mass_var.get()

        self.current_params = {'nx': nx, 'ny': ny, 'omega': omega, 'mass': mass}
        
        #calc energy level
        energy = hbar * omega * (nx + ny + 1)
        self.energy_var.set(f"Energy: E = {energy:.2f} ℏω")
        
        #state description based on quantum numbers
        if nx == 0 and ny == 0:
            state_desc = "Ground State (lowest energy)"
        elif (nx == 1 and ny == 0) or (nx == 0 and ny == 1):
            state_desc = "First Excited State"
        elif nx == 1 and ny == 1:
            state_desc = "Doubly Excited State"
        else:
            state_desc = f"Excited State (nx={nx}, ny={ny})"
        self.state_var.set(state_desc)
        
        #wavefunction and probability density
        wave = wavefunction(nx, ny, self.X, self.Y, omega, mass)
        prob = np.abs(wave)**2
        
        if self.contour_plot is not None:
            try:
                for coll in self.contour_plot.collections:
                    if coll in self.ax.collections:
                        coll.remove()
            except:
                self.ax.clear()
                self.ax.set_facecolor('#222222')
                self.ax.grid(True, linestyle='--', alpha=0.3)
                self.ax.axhline(y=0, color='white', linestyle='-', alpha=0.2)
                self.ax.axvline(x=0, color='white', linestyle='-', alpha=0.2)
                self.ax.set_xlabel("Position x", color='white', fontsize=12)
                self.ax.set_ylabel("Position y", color='white', fontsize=12)
                self.ax.tick_params(colors='white')
                
                try:
                    cmap = "quantum"
                    self.im = self.ax.imshow(prob, extent=(-5, 5, -5, 5),
                                         cmap=cmap, origin="lower", aspect='equal')
                except:
                    cmap = "viridis"
                    self.im = self.ax.imshow(prob, extent=(-5, 5, -5, 5),
                                         cmap=cmap, origin="lower", aspect='equal')
        
        potential = 0.5 * mass * omega**2 * (self.X**2 + self.Y**2)
        self.contour_plot = self.ax.contour(self.X, self.Y, potential, 
                                           levels=np.linspace(0, 10*omega*mass, 10), 
                                           colors='white', alpha=0.3, linewidths=0.5)

        if 'im' in self.__dict__ and self.im in self.ax.images:
            self.im.set_data(prob)
            self.im.set_clim(vmin=0, vmax=np.max(prob))
        
        self.ax.set_title(f"Quantum Harmonic Oscillator - State |{nx},{ny}⟩", 
                         color='white', fontsize=14)
        
        self.canvas.draw_idle()

if __name__ == "__main__":
    root = tk.Tk()
    app = QuantumHarmonicOscillatorApp(root)
    root.mainloop()