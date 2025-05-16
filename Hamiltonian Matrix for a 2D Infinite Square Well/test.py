import numpy as np
import tkinter as tk
from tkinter import ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import eigsh
from matplotlib import cm
import matplotlib as mpl

hbar = 1.0  #reduced Planck constant (in atomic units)

DARK_BG = "#1e1e2e"  
LIGHT_TEXT = "#cdd6f4" 
ACCENT_1 = "#89b4fa"
ACCENT_2 = "#f38ba8"  
ACCENT_3 = "#a6e3a1"
PANEL_BG = "#313244"

def create_hamiltonian(L_x, L_y, m, N):
    """
    Create the Hamiltonian matrix for a 2D infinite square well.
    Uses finite difference method to discretize the Laplacian operator.
    
    Args:
        L_x: Width of potential well
        L_y: Height of potential well
        m: Particle mass
        N: Number of grid points in each dimension
    
    Returns:
        Sparse Hamiltonian matrix
    """
    #spatial grid
    dx = L_x / (N + 1)
    dy = L_y / (N + 1)
    
    #prefactor for kinetic energy: -ħ²/(2m)
    prefactor = -hbar**2 / (2 * m)
    
    #create 1D Laplacian operators for x and y directions
    diag = np.ones(N)
    diags = np.array([1, -2, 1])  #finite difference stencil
    offsets = np.array([-1, 0, 1])
    laplacian_1d = sparse.spdiags(np.vstack([diag, -2*diag, diag]), offsets, N, N)

    laplacian_x = laplacian_1d / (dx**2)
    laplacian_y = laplacian_1d / (dy**2)
    
    #2D Laplacian using Kronecker products
    identity = sparse.eye(N)
    laplacian_2d = sparse.kron(laplacian_x, identity) + sparse.kron(identity, laplacian_y)
    
    #compute Hamiltonian: -ħ²/(2m) ∇²
    H = prefactor * laplacian_2d
    
    return H

def solve_schrodinger_equation(L_x, L_y, m, n_states=5, grid_size=50):
    """
    Numerically solve the time-independent Schrödinger equation for a 2D infinite square well.
    
    Args:
        L_x: Width of potential well
        L_y: Height of potential well
        m: Particle mass
        n_states: Number of eigenstates to compute
        grid_size: Number of grid points in each dimension
    
    Returns:
        energies: Eigenvalues (energies)
        states: Eigenvectors (wavefunctions)
        x_grid, y_grid: Coordinate grids
    """
    #create Hamiltonian matrix
    H = create_hamiltonian(L_x, L_y, m, grid_size)
    
    #Solve eigenvalue problem for lowest n_states eigenstates
    energies, states = eigsh(H, k=n_states, which='SM')
    
    #Reshape eigenvectors to 2D grids
    shaped_states = [state.reshape(grid_size, grid_size) for state in states.T]

    x = np.linspace(0, L_x, grid_size)
    y = np.linspace(0, L_y, grid_size)
    x_grid, y_grid = np.meshgrid(x, y)
    
    return energies, shaped_states, x_grid, y_grid

def get_analytical_state(n_x, n_y, L_x, L_y, x_grid, y_grid):
    """
    Calculate the analytical wavefunction for given quantum numbers.
    
    Args:
        n_x, n_y: Quantum numbers
        L_x, L_y: Box dimensions
        x_grid, y_grid: Coordinate grids
    
    Returns:
        Probability density
    """
    #calc normalized wave function
    wave_function = np.sqrt(4/(L_x*L_y)) * np.sin(n_x*np.pi*x_grid/L_x) * np.sin(n_y*np.pi*y_grid/L_y)
    
    #calc probability density
    probability_density = np.abs(wave_function)**2
    
    return probability_density

def calculate_analytical_energy(n_x, n_y, L_x, L_y, m):
    """
    Calculate the energy eigenvalue for a 2D infinite square well using the analytical formula.
    
    Args:
        n_x, n_y: Quantum numbers
        L_x, L_y: Box dimensions
        m: Particle mass
    
    Returns:
        Energy eigenvalue
    """
    return (hbar**2 * np.pi**2) / (2 * m) * ((n_x/L_x)**2 + (n_y/L_y)**2)

def setup_styles():
    style = ttk.Style()
    style.theme_use('clam')

    style.configure("TFrame", background=DARK_BG)
    style.configure("TLabel", background=DARK_BG, foreground=LIGHT_TEXT, font=('Helvetica', 10))
    style.configure("TButton", background=PANEL_BG, foreground=LIGHT_TEXT)
    style.configure("TCheckbutton", background=DARK_BG, foreground=LIGHT_TEXT)
    style.configure("TRadiobutton", background=DARK_BG, foreground=LIGHT_TEXT, font=('Helvetica', 10))

    style.configure("energy.Horizontal.TProgressbar", 
                   background=ACCENT_2, troughcolor=PANEL_BG)
    
    style.configure("TScale", background=DARK_BG, troughcolor=PANEL_BG, sliderrelief="flat")
    
    style.configure("Control.TFrame", background=PANEL_BG, relief="raised", borderwidth=1)
    
    style.configure("TSeparator", background=LIGHT_TEXT)

def custom_slider(parent, from_, to_, resolution, label_text, initial, command, width=None):
    """Create a custom styled slider with label"""
    if width is None:
        width = sidebar_width - 60
    
    frame = ttk.Frame(parent, style="Control.TFrame")
    
    header_frame = ttk.Frame(frame, style="Control.TFrame")
    header_frame.pack(fill=tk.X, padx=5, pady=(5, 0))

    label = ttk.Label(header_frame, text=label_text, style="TLabel", font=('Helvetica', 9))
    label.pack(side=tk.LEFT, pady=2)
    
    value_var = tk.StringVar(value=f"{initial:.1f}" if resolution != 1 else f"{int(initial)}")
    
    value_label = ttk.Label(header_frame, textvariable=value_var, width=5, 
                           font=('Helvetica', 8, 'bold'), foreground=ACCENT_1)
    value_label.pack(side=tk.RIGHT, pady=2)

    def on_slider_change(event=None):
        val = slider.get()
        if resolution == 1:
            value_var.set(f"{int(val)}")
        else:
            value_var.set(f"{val:.1f}")
        command()

    slider = ttk.Scale(frame, from_=from_, to=to_, length=width)
    slider.set(initial)
    slider.bind("<ButtonRelease-1>", on_slider_change)
    slider.bind("<Motion>", lambda e: slider.get())
    slider.pack(padx=5, pady=(0, 5))
    
    return frame, slider

root = tk.Tk()
root.title("Quantum Particle in a Box Simulator")

screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

window_width = min(int(screen_width * 0.8), 1200)
window_height = min(int(screen_height * 0.8), 800)

x_position = (screen_width - window_width) // 2
y_position = (screen_height - window_height) // 2
root.geometry(f"{window_width}x{window_height}+{x_position}+{y_position}")

root.configure(bg=DARK_BG)
root.option_add("*Font", "Helvetica 10")

root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)

setup_styles()

main_frame = ttk.Frame(root, style="TFrame")
main_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
main_frame.columnconfigure(0, weight=1)
main_frame.rowconfigure(1, weight=1)

header_frame = ttk.Frame(main_frame, style="TFrame")
header_frame.pack(fill=tk.X, pady=(0, 10))

title_label = ttk.Label(header_frame, text="Quantum Particle in a 2D Box", 
                        font=('Helvetica', 18, 'bold'), foreground=ACCENT_1)
title_label.pack(side=tk.LEFT, padx=10)

content_frame = ttk.Frame(main_frame, style="TFrame")
content_frame.pack(fill=tk.BOTH, expand=True)

sidebar_width = min(300, int(window_width * 0.25))

content_frame.columnconfigure(0, minsize=sidebar_width, weight=0)
content_frame.columnconfigure(1, weight=1)
content_frame.rowconfigure(0, weight=1)

sidebar_frame = ttk.Frame(content_frame, style="Control.TFrame", width=sidebar_width)
sidebar_frame.grid(row=0, column=0, sticky="ns", padx=(0, 10), pady=5)
sidebar_frame.grid_propagate(False)

canvas_controls = tk.Canvas(sidebar_frame, bg=PANEL_BG, highlightthickness=0)
scrollbar = ttk.Scrollbar(sidebar_frame, orient="vertical", command=canvas_controls.yview)
scrollable_frame = ttk.Frame(canvas_controls, style="Control.TFrame")

scrollable_frame.bind(
    "<Configure>",
    lambda e: canvas_controls.configure(scrollregion=canvas_controls.bbox("all"))
)

canvas_controls.create_window((0, 0), window=scrollable_frame, anchor="nw")
canvas_controls.configure(yscrollcommand=scrollbar.set)

canvas_controls.pack(side="left", fill="both", expand=True)
scrollbar.pack(side="right", fill="y")

plot_frame = ttk.Frame(content_frame, style="TFrame")
plot_frame.grid(row=0, column=1, sticky="nsew")
plot_frame.columnconfigure(0, weight=1)
plot_frame.rowconfigure(0, weight=1)

# ===== Left Sidebar Controls - now placed in scrollable_frame =====
method_frame = ttk.Frame(scrollable_frame, style="Control.TFrame")
method_frame.pack(fill=tk.X, padx=10, pady=10)

method_label = ttk.Label(method_frame, text="Solution Method", font=('Helvetica', 10, 'bold'), foreground=ACCENT_1)
method_label.pack(pady=(5, 10))

method_var = tk.StringVar(value="Analytical")
analytical_rb = ttk.Radiobutton(method_frame, text="Analytical Solution", 
                               variable=method_var, value="Analytical")
analytical_rb.pack(fill=tk.X, padx=20, pady=2)

numerical_rb = ttk.Radiobutton(method_frame, text="Numerical PDE Solver", 
                              variable=method_var, value="Numerical")
numerical_rb.pack(fill=tk.X, padx=20, pady=2)

quantum_frame = ttk.Frame(scrollable_frame, style="Control.TFrame")
quantum_frame.pack(fill=tk.X, padx=10, pady=(5, 10))

quantum_label = ttk.Label(quantum_frame, text="Quantum Numbers", font=('Helvetica', 10, 'bold'), foreground=ACCENT_1)
quantum_label.pack(pady=(5, 10))

quantum_display = tk.Canvas(quantum_frame, width=sidebar_width-40, height=60, bg=PANEL_BG, highlightthickness=0)
quantum_display.pack(pady=(0, 5))

display_width = sidebar_width-40
quantum_display.create_rectangle(10, 10, display_width-10, 50, outline=ACCENT_1, width=2)
quantum_display.create_text(display_width//2, 5, text="Quantum State", fill=LIGHT_TEXT)
nx_text = quantum_display.create_text(display_width//3, 30, text="1", fill=ACCENT_2, font=('Helvetica', 16, 'bold'))
ny_text = quantum_display.create_text(2*display_width//3, 30, text="1", fill=ACCENT_3, font=('Helvetica', 16, 'bold'))
quantum_display.create_text(display_width//3, 50, text="nx", fill=ACCENT_2)
quantum_display.create_text(2*display_width//3, 50, text="ny", fill=ACCENT_3)
quantum_display.create_text(display_width//2, 30, text="×", fill=LIGHT_TEXT)

def update_plot(*args):
    pass

n_x_frame, n_x_slider = custom_slider(quantum_frame, 1, 5, 1, "Quantum Number nx:", 1, update_plot)
n_x_frame.pack(fill=tk.X, pady=2)

n_y_frame, n_y_slider = custom_slider(quantum_frame, 1, 5, 1, "Quantum Number ny:", 1, update_plot)
n_y_frame.pack(fill=tk.X, pady=2)

box_frame = ttk.Frame(scrollable_frame, style="Control.TFrame")
box_frame.pack(fill=tk.X, padx=10, pady=5)

box_label = ttk.Label(box_frame, text="Box Properties", font=('Helvetica', 10, 'bold'), foreground=ACCENT_1)
box_label.pack(pady=(5, 5))

L_x_frame, L_x_slider = custom_slider(box_frame, 1, 10, 0.1, "Box Width (Lx):", 5.0, update_plot)
L_x_frame.pack(fill=tk.X, pady=2)

L_y_frame, L_y_slider = custom_slider(box_frame, 1, 10, 0.1, "Box Height (Ly):", 5.0, update_plot)
L_y_frame.pack(fill=tk.X, pady=2)

m_frame, m_slider = custom_slider(box_frame, 0.1, 5.0, 0.1, "Particle Mass (m):", 1.0, update_plot)
m_frame.pack(fill=tk.X, pady=2)

visual_frame = ttk.Frame(scrollable_frame, style="Control.TFrame")
visual_frame.pack(fill=tk.X, padx=10, pady=(5, 10))

visual_label = ttk.Label(visual_frame, text="Visualization Options", font=('Helvetica', 10, 'bold'), foreground=ACCENT_1)
visual_label.pack(pady=(5, 5))

show_nodal_lines = tk.BooleanVar(value=True)
nodal_check = ttk.Checkbutton(visual_frame, text="Show Nodal Lines", variable=show_nodal_lines)
nodal_check.pack(fill=tk.X, padx=20, pady=2)

energy_frame = ttk.Frame(scrollable_frame, style="Control.TFrame")
energy_frame.pack(fill=tk.X, padx=10, pady=5)

energy_header = ttk.Label(energy_frame, text="Energy", font=('Helvetica', 10, 'bold'), foreground=ACCENT_1)
energy_header.pack(pady=(5, 5))

energy_meter = ttk.Progressbar(energy_frame, style="energy.Horizontal.TProgressbar", 
                             orient="horizontal", mode="determinate", value=50)
energy_meter.pack(fill=tk.X, padx=10, pady=5)

energy_display_frame = ttk.Frame(energy_frame, style="Control.TFrame")
energy_display_frame.pack(fill=tk.X, padx=10, pady=2)

analytical_label = ttk.Label(energy_display_frame, text="Analytical:")
analytical_label.grid(row=0, column=0, sticky="w", padx=5)

energy_label = ttk.Label(energy_display_frame, text="0.0000", foreground=ACCENT_2, font=('Helvetica', 10, 'bold'))
energy_label.grid(row=0, column=1, sticky="w", padx=5)

numerical_energy_label_text = ttk.Label(energy_display_frame, text="Numerical:")
numerical_energy_label_text.grid(row=1, column=0, sticky="w", padx=5)

numerical_energy_label = ttk.Label(energy_display_frame, text="0.0000", foreground=ACCENT_3, font=('Helvetica', 10, 'bold'))
numerical_energy_label.grid(row=1, column=1, sticky="w", padx=5)

equation_frame = ttk.Frame(scrollable_frame, style="Control.TFrame")
equation_frame.pack(fill=tk.X, padx=10, pady=(5, 10))

equation_header = ttk.Label(equation_frame, text="Energy Equation", font=('Helvetica', 10, 'bold'), foreground=ACCENT_1)
equation_header.pack(pady=(5, 2))

equation_label = ttk.Label(equation_frame, text="E = (π²ħ²/2m)[(nx/Lx)² + (ny/Ly)²]", 
                          font=('Courier', 9), wraplength=sidebar_width-30)
equation_label.pack(pady=(2, 5))

plot_container = ttk.Frame(plot_frame, style="Control.TFrame")
plot_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
plot_container.columnconfigure(0, weight=1)
plot_container.rowconfigure((0, 1), weight=1)

plt.style.use('dark_background')
mpl.rcParams['axes.facecolor'] = PANEL_BG
mpl.rcParams['figure.facecolor'] = PANEL_BG
mpl.rcParams['savefig.facecolor'] = PANEL_BG

fig = Figure(figsize=(6, 4), dpi=100, facecolor=PANEL_BG)
ax = fig.add_subplot(111)
ax.set_facecolor(PANEL_BG)
ax.set_aspect('equal', adjustable='box')

figure_frame = ttk.Frame(plot_container, style="Control.TFrame")
figure_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 5))
figure_frame.columnconfigure(0, weight=1)
figure_frame.rowconfigure(0, weight=1)

canvas = FigureCanvasTkAgg(fig, master=figure_frame)
canvas.draw()
canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

#Create 3D visualization with fixed size
fig3d = Figure(figsize=(6, 3), dpi=100, facecolor=PANEL_BG)
ax3d = fig3d.add_subplot(111, projection='3d')
ax3d.set_facecolor(PANEL_BG)

ax3d.view_init(elev=30, azim=45)

figure3d_frame = ttk.Frame(plot_container, style="Control.TFrame")
figure3d_frame.grid(row=1, column=0, sticky="nsew")
figure3d_frame.columnconfigure(0, weight=1)
figure3d_frame.rowconfigure(0, weight=1)

canvas3d = FigureCanvasTkAgg(fig3d, master=figure3d_frame)
canvas3d.draw()
canvas3d.get_tk_widget().grid(row=0, column=0, sticky="nsew")

def on_resize(event):
    canvas.get_tk_widget().configure(width=event.width-20, height=event.height//2-20)
    canvas3d.get_tk_widget().configure(width=event.width-20, height=event.height//2-20)
    
    canvas.draw()
    canvas3d.draw()

plot_container.bind("<Configure>", on_resize)

physics_frame = ttk.Frame(main_frame, style="Control.TFrame")
physics_frame.pack(fill=tk.X, pady=(10, 0))

explanation = """
The time-independent Schrödinger equation for a 2D particle in a box is:  [-ħ²/(2m)]∇²ψ = Eψ
Analytical solution: ψ(x,y) = √(4/LₓLᵧ) · sin(nₓπx/Lₓ) · sin(nᵧπy/Lᵧ)
Energy levels: E(nₓ,nᵧ) = (ħ²π²/2m)[(nₓ/Lₓ)² + (nᵧ/Lᵧ)²]
"""

physics_label = ttk.Label(physics_frame, text=explanation, font=('Courier', 9), justify=tk.LEFT)
physics_label.pack(pady=10, padx=10)

def update_plot(*args):
    """Update the plots based on slider values."""
    n_x = int(n_x_slider.get())
    n_y = int(n_y_slider.get())
    L_x = L_x_slider.get()
    L_y = L_y_slider.get()
    m = m_slider.get()
    method = method_var.get()
    cmap_name = "viridis"
    
    energy_val = calculate_analytical_energy(n_x, n_y, L_x, L_y, m)
    energy_label.config(text=f"{energy_val:.4f}")
    
    energy_level = min(energy_val / 20.0, 1.0)  # Scale to max of 20
    energy_meter.config(value=energy_level * 100)
    
    if hasattr(update_plot, 'colorbar') and update_plot.colorbar is not None:
        try:
            update_plot.colorbar.remove()
        except:
            pass

    if method == "Analytical":
        x = np.linspace(0, L_x, 150)
        y = np.linspace(0, L_y, 150)
        X, Y = np.meshgrid(x, y)
        probability_density = get_analytical_state(n_x, n_y, L_x, L_y, X, Y)

        ax.clear()

        im = ax.imshow(probability_density, cmap=cmap_name, interpolation='bilinear',
                  extent=[0, L_x, 0, L_y], origin='lower', aspect='equal')
        
        ax.set_title(f"Probability Density", color='white', fontsize=12)
        ax.set_xlabel(f"x (L_x = {L_x:.1f})", color='white')
        ax.set_ylabel(f"y (L_y = {L_y:.1f})", color='white')
        ax.tick_params(colors='white')
        
        ax3d.clear()
        X, Y = np.meshgrid(x, y)

        X_norm = X / L_x
        Y_norm = Y / L_y
        
        surf = ax3d.plot_surface(X_norm, Y_norm, probability_density, cmap=cmap_name, 
                              antialiased=True, alpha=0.8)

        ax3d.set_xlim(0, 1)
        ax3d.set_ylim(0, 1)
        ax3d.set_zlim(0, np.max(probability_density) * 1.1)
        
        ax3d.set_title(f"3D Visualization (nx={n_x}, ny={n_y})", color='white', fontsize=10)
        ax3d.set_xlabel(f"x/L_x", color='white')
        ax3d.set_ylabel(f"y/L_y", color='white')
        ax3d.set_zlabel("|ψ|²", color='white')
        ax3d.tick_params(colors='white')
        
    else:  #"Numerical"
        #Use numerical PDE solver
        grid_size = 60  #higher resolution
        energies, states, x_grid, y_grid = solve_schrodinger_equation(L_x, L_y, m, n_states=max(n_x*n_y, 5), grid_size=grid_size)
        
        #find state matching desired quantum numbers (approximately)
        state_idx = min(n_x * n_y - 1, len(states) - 1)
        probability_density = np.abs(states[state_idx])**2
        
        #normalize for visualization
        probability_density /= np.max(probability_density)
        
        ax.clear()
        
        extent_vals = [0, L_x, 0, L_y]
        im = ax.imshow(probability_density, cmap=cmap_name, interpolation='bilinear',
                  extent=extent_vals, origin='lower', aspect='equal')
                  
        max_dim = max(L_x, L_y)
        padding = max_dim * 0.1
        ax.set_xlim(-padding, L_x + padding)
        ax.set_ylim(-padding, L_y + padding)
        
        ax.set_title(f"Probability Density", color='white', fontsize=12)
        ax.set_xlabel(f"x (L_x = {L_x:.1f})", color='white')
        ax.set_ylabel(f"y (L_y = {L_y:.1f})", color='white')
        ax.tick_params(colors='white')
        
        numerical_energy_label.config(text=f"{energies[state_idx]:.4f}")

        ax3d.clear()
        
        X_norm = x_grid / L_x
        Y_norm = y_grid / L_y
        
        surf = ax3d.plot_surface(X_norm, Y_norm, probability_density, cmap=cmap_name, 
                              antialiased=True, alpha=0.8)
        
        ax3d.set_xlim(0, 1)
        ax3d.set_ylim(0, 1)
        ax3d.set_zlim(0, np.max(probability_density) * 1.1)
        
        ax3d.set_title(f"3D Visualization (State {state_idx+1})", color='white', fontsize=10)
        ax3d.set_xlabel(f"x/L_x", color='white')
        ax3d.set_ylabel(f"y/L_y", color='white')
        ax3d.set_zlabel("|ψ|²", color='white')
        ax3d.tick_params(colors='white')

    update_plot.colorbar = fig.colorbar(im, ax=ax)
    update_plot.colorbar.ax.tick_params(colors='white')

    if show_nodal_lines.get():
        for i in range(1, n_x):
            x_pos = i * L_x / n_x
            ax.axvline(x=x_pos, color='white', linestyle='--', alpha=0.5, linewidth=0.7)
        
        for j in range(1, n_y):
            y_pos = j * L_y / n_y
            ax.axhline(y=y_pos, color='white', linestyle='--', alpha=0.5, linewidth=0.7)

    fig.subplots_adjust(left=0.15, right=0.85, bottom=0.15, top=0.9)
    fig3d.subplots_adjust(left=0.15, right=0.85, bottom=0.15, top=0.9)
    
    canvas.draw()
    canvas3d.draw()

    quantum_display.itemconfig(nx_text, text=f"{n_x}")
    quantum_display.itemconfig(ny_text, text=f"{n_y}")

    eqn_text = f"E = {energy_val:.2f} = (π²ħ²/2m)[({n_x}/{L_x:.1f})² + ({n_y}/{L_y:.1f})²]"
    equation_label.config(text=eqn_text)

analytical_rb.config(command=update_plot)
numerical_rb.config(command=update_plot)
nodal_check.config(command=update_plot)

def bind_slider_events(slider, command):
    slider.bind("<B1-Motion>", lambda e: command())
    slider.bind("<ButtonRelease-1>", lambda e: command())

bind_slider_events(n_x_slider, update_plot)
bind_slider_events(n_y_slider, update_plot)
bind_slider_events(L_x_slider, update_plot)
bind_slider_events(L_y_slider, update_plot)
bind_slider_events(m_slider, update_plot)

update_plot()

root.mainloop()