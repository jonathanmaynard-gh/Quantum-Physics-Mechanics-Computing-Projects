import numpy as np
import matplotlib.pyplot as plt
from scipy.special import hermite, factorial
import matplotlib.animation as animation
from matplotlib.widgets import Slider, RadioButtons, Button

#constants
hbar = 1.0  #Reduced Planck constant
m = 1.0     #Mass
omega = 1.0 #Angular frequency

#calc Hermite polynomials
def hermite_polynomial(n, x):
    return hermite(n)(x)

#1D wavefunction
def wavefunction_1d(n, x):
    """Calculate the 1D quantum harmonic oscillator wavefunction for state n."""
    #Normalization constant
    norm = 1.0 / np.sqrt(2**n * np.sqrt(np.pi) * factorial(n))
    return norm * np.exp(-x**2/2) * hermite_polynomial(n, x)

#2D wavefunction with separate quantum numbers
def wavefunction_2d(nx, ny, x, y):
    """
    Calculate the 2D quantum harmonic oscillator wavefunction for state (nx, ny).
    
    Parameters:
    -----------
    nx, ny : int
        Quantum numbers for x and y directions
    x, y : array
        Position coordinates
    
    Returns:
    --------
    psi : array
        The 2D wavefunction
    """
    psi_x = wavefunction_1d(nx, x)
    psi_y = wavefunction_1d(ny, y)
    return psi_x[:, np.newaxis] * psi_y[np.newaxis, :]

def probability_density(nx, ny, x, y):
    """Calculate the probability density |ψ|² for state (nx, ny)."""
    psi = wavefunction_2d(nx, ny, x, y)
    return np.abs(psi)**2

#superposition of states
def wavefunction_superposition(nx1, ny1, nx2, ny2, x, y, t, weight1=0.5, weight2=0.5):
    """
    Calculate a time-dependent superposition of two energy eigenstates.
    
    Parameters:
    -----------
    nx1, ny1 : int
        Quantum numbers of first state
    nx2, ny2 : int
        Quantum numbers of second state
    x, y : array
        Position coordinates
    t : float
        Time
    weight1, weight2 : float
        Relative weights of states in superposition
    
    Returns:
    --------
    psi : complex array
        The superposition wavefunction
    """
    #Energy of each state: E = hbar * omega * (nx + ny + 1)
    E1 = hbar * omega * (nx1 + ny1 + 1)
    E2 = hbar * omega * (nx2 + ny2 + 1)
    
    #Time evolution factors
    phase1 = np.exp(-1j * E1 * t / hbar)
    phase2 = np.exp(-1j * E2 * t / hbar)
    
    #Normalized superposition
    norm_factor = np.sqrt(weight1**2 + weight2**2)
    psi = (weight1 * wavefunction_2d(nx1, ny1, x, y) * phase1 + 
           weight2 * wavefunction_2d(nx2, ny2, x, y) * phase2) / norm_factor
    
    return psi

x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)

fig = plt.figure(figsize=(12, 10))
plt.subplots_adjust(left=0.1, bottom=0.35, right=0.9, top=0.9)

ax = fig.add_subplot(111)
ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('y', fontsize=12)
ax.grid(True, alpha=0.3)

#init states and times
nx_state = 0
ny_state = 0
time_val = 0.0
is_superposition = False
nx1 = 0
ny1 = 0
nx2 = 1
ny2 = 0
weight1 = 0.7
weight2 = 0.3

#calc init probability density
if is_superposition:
    Z = np.abs(wavefunction_superposition(
        nx1, ny1, nx2, ny2, x, y, time_val, weight1, weight2))**2
else:
    Z = probability_density(nx_state, ny_state, x, y)

img = ax.imshow(Z, cmap='viridis', extent=(x.min(), x.max(), y.min(), y.max()), origin='lower')
title = ax.set_title(f"Probability Density for (nx={nx_state}, ny={ny_state})", fontsize=14)
cbar = plt.colorbar(img, ax=ax, label="Probability Density")

ax_nx = plt.axes([0.1, 0.25, 0.35, 0.03])
ax_ny = plt.axes([0.55, 0.25, 0.35, 0.03])
ax_time = plt.axes([0.1, 0.20, 0.8, 0.03])
ax_nx1 = plt.axes([0.1, 0.15, 0.35, 0.03])
ax_ny1 = plt.axes([0.55, 0.15, 0.35, 0.03])
ax_nx2 = plt.axes([0.1, 0.10, 0.35, 0.03])
ax_ny2 = plt.axes([0.55, 0.10, 0.35, 0.03])
ax_weight = plt.axes([0.1, 0.05, 0.8, 0.03])

ax_radio = plt.axes([0.02, 0.4, 0.15, 0.15])
radio = RadioButtons(ax_radio, ('Single State', 'Superposition'))

ax_play = plt.axes([0.02, 0.7, 0.1, 0.05])
ax_pause = plt.axes([0.13, 0.7, 0.1, 0.05])
ax_reset = plt.axes([0.02, 0.6, 0.1, 0.05])

play_button = Button(ax_play, 'Play')
pause_button = Button(ax_pause, 'Pause')
reset_button = Button(ax_reset, 'Reset')

nx_slider = Slider(ax_nx, 'Quantum Number nx', 0, 5, valinit=nx_state, valstep=1)
ny_slider = Slider(ax_ny, 'Quantum Number ny', 0, 5, valinit=ny_state, valstep=1)
time_slider = Slider(ax_time, 'Time', 0, 10, valinit=time_val)
nx1_slider = Slider(ax_nx1, 'State 1: nx', 0, 5, valinit=nx1, valstep=1)
ny1_slider = Slider(ax_ny1, 'State 1: ny', 0, 5, valinit=ny1, valstep=1)
nx2_slider = Slider(ax_nx2, 'State 2: nx', 0, 5, valinit=nx2, valstep=1)
ny2_slider = Slider(ax_ny2, 'State 2: ny', 0, 5, valinit=ny2, valstep=1)
weight_slider = Slider(ax_weight, 'State 1 Weight', 0, 1, valinit=weight1)

ani = None
is_animating = False

def update_plot(val=None):
    """Update the plot based on current settings."""
    global Z
    
    if is_superposition:
        Z = np.abs(wavefunction_superposition(
            int(nx1_slider.val), int(ny1_slider.val),
            int(nx2_slider.val), int(ny2_slider.val),
            x, y, time_slider.val, 
            weight_slider.val, 1 - weight_slider.val))**2
        
        e1 = hbar * omega * (int(nx1_slider.val) + int(ny1_slider.val) + 1)
        e2 = hbar * omega * (int(nx2_slider.val) + int(ny2_slider.val) + 1)
        
        title_text = (f"Superposition of (nx,ny)=({int(nx1_slider.val)},{int(ny1_slider.val)}) "
                     f"and ({int(nx2_slider.val)},{int(ny2_slider.val)}), t={time_slider.val:.2f}\n"
                     f"E₁={e1:.2f}, E₂={e2:.2f}, ΔE={abs(e2-e1):.2f}")
    else:
        nx = int(nx_slider.val)
        ny = int(ny_slider.val)
        Z = probability_density(nx, ny, x, y)
        
        energy = hbar * omega * (nx + ny + 1)
        title_text = f"Probability Density for (nx={nx}, ny={ny}), E={energy:.2f}"
    
    img.set_data(Z)
    img.set_clim(Z.min(), Z.max())
    title.set_text(title_text)
    fig.canvas.draw_idle()

def toggle_mode(label):
    """Toggle between single state and superposition modes."""
    global is_superposition
    if label == 'Single State':
        is_superposition = False
        ax_nx.set_visible(True)
        ax_ny.set_visible(True)
        ax_nx1.set_visible(False)
        ax_ny1.set_visible(False)
        ax_nx2.set_visible(False)
        ax_ny2.set_visible(False)
        ax_weight.set_visible(False)
    else:
        is_superposition = True
        ax_nx.set_visible(False)
        ax_ny.set_visible(False)
        ax_nx1.set_visible(True)
        ax_ny1.set_visible(True)
        ax_nx2.set_visible(True)
        ax_ny2.set_visible(True)
        ax_weight.set_visible(True)
    update_plot()
    fig.canvas.draw_idle()

def init_animation():
    """Initialize the animation."""
    return [img]

def animate(i):
    """Update function for animation frames."""
    time_val = (time_slider.val + 0.1) % 10
    time_slider.set_val(time_val)
    return [img]

def play_animation(event):
    """Start the animation."""
    global ani, is_animating
    if not is_animating:
        ani = animation.FuncAnimation(
            fig, animate, init_func=init_animation,
            frames=100, interval=100, blit=True)
        is_animating = True

def pause_animation(event):
    """Pause the animation."""
    global ani, is_animating
    if is_animating and ani is not None:
        ani.event_source.stop()
        is_animating = False

def reset_animation(event):
    """Reset the animation to time=0."""
    global ani, is_animating
    if is_animating and ani is not None:
        ani.event_source.stop()
        is_animating = False
    time_slider.set_val(0)

nx_slider.on_changed(update_plot)
ny_slider.on_changed(update_plot)
time_slider.on_changed(update_plot)
nx1_slider.on_changed(update_plot)
ny1_slider.on_changed(update_plot)
nx2_slider.on_changed(update_plot)
ny2_slider.on_changed(update_plot)
weight_slider.on_changed(update_plot)
radio.on_clicked(toggle_mode)
play_button.on_clicked(play_animation)
pause_button.on_clicked(pause_animation)
reset_button.on_clicked(reset_animation)

ax_nx1.set_visible(False)
ax_ny1.set_visible(False)
ax_nx2.set_visible(False)
ax_ny2.set_visible(False)
ax_weight.set_visible(False)

plt.figtext(0.02, 0.01, 
            "2D Quantum Harmonic Oscillator\n"
            "E = ħω(nx + ny + 1)\n"
            "Wavefunction: ψ(x,y) = ψ_nx(x) × ψ_ny(y)",
            fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

plt.show()