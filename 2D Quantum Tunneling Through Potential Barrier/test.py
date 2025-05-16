import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider
import matplotlib.patches as patches

#physical constants
hbar = 1.0  #Reduced Planck constant (ħ) in natural units
m = 1.0  #particle mass in natural units

#grid parameters
Nx, Ny = 100, 100 
x_min, x_max = -10.0, 10.0
y_min, y_max = -10.0, 10.0
dx = (x_max - x_min) / (Nx - 1)
dy = (y_max - y_min) / (Ny - 1)
x = np.linspace(x_min, x_max, Nx)
y = np.linspace(y_min, y_max, Ny)
X, Y = np.meshgrid(x, y)

# ime parameters (reduced dt and increased frames for smoother animation)
dt = 0.03  #Time step (reduced for better time resolution)
frames = 100  #Number of frames (increased for smoother animation)

#init parameters
barrier_height = 2.0
barrier_width = 2.0
barrier_position = 0.0
energy = 1.0  #Initial energy
initial_position = -5.0  #Initial position
sigma = 0.8  #Width of wave packet

#Precalculate initial momentum
initial_momentum = np.sqrt(2 * m * energy)

def create_potential(barrier_height=1.0, barrier_width=2.0, barrier_position=0.0):
    """
    Create a 2D potential barrier.
    
    This creates a rectangular potential barrier spanning across the y-axis,
    with specified height, width, and position along the x-axis.
    """
    V = np.zeros((Ny, Nx))
    V[:, ((X[0, :] >= barrier_position - barrier_width/2) & 
          (X[0, :] <= barrier_position + barrier_width/2))] = barrier_height
    return V

V = create_potential(barrier_height, barrier_width, barrier_position)

def create_wave_packet(x0=initial_position, k0=initial_momentum, sigma=sigma):
    """
    Create a 2D Gaussian wave packet.
    
    This constructs a Gaussian wave packet centered at position x0 with 
    momentum k0 (wave number) and width sigma. The wave packet is normalized
    to ensure total probability equals 1.
    
    The wave function has form: ψ(x,y) = A * exp(-(x-x0)²/2σ² - y²/2σ²) * exp(ik0*x)
    """
    psi = np.exp(-((X - x0)**2 + Y**2) / (2 * sigma**2)) * np.exp(1j * k0 * X)
    norm = np.sqrt(np.sum(np.abs(psi)**2) * dx * dy)
    return psi / norm

psi = create_wave_packet()

#Time evolution with split-step Fourier method
def evolve_wavefunction(psi, V, dt):
    """
    Evolve the wave function one time step using the split-step Fourier method.
    
    This method solves the time-dependent Schrödinger equation:
    iħ∂ψ/∂t = [-ħ²/2m ∇² + V(x,y)]ψ
    
    The split-step method works by:
    1. Apply half-step of potential operator: exp(-iVdt/2ħ)
    2. Apply full-step of kinetic operator in Fourier space: exp(-iħk²dt/2m)
    3. Apply another half-step of potential operator
    
    This is second-order accurate in time and numerically stable.
    """
    #Potential part (half step)
    psi = psi * np.exp(-1j * V * dt / (2 * hbar))
    
    #Kinetic part (x-direction)
    psi_k = np.fft.fft(psi, axis=1)
    kx = 2 * np.pi * np.fft.fftfreq(Nx, dx)
    psi_k = psi_k * np.exp(-1j * hbar * kx[np.newaxis, :]**2 * dt / (2 * m))
    psi = np.fft.ifft(psi_k, axis=1)
    
    #Kinetic part (y-direction)
    psi_k = np.fft.fft(psi, axis=0)
    ky = 2 * np.pi * np.fft.fftfreq(Ny, dy)
    psi_k = psi_k * np.exp(-1j * hbar * ky[:, np.newaxis]**2 * dt / (2 * m))
    psi = np.fft.ifft(psi_k, axis=0)
    
    #Potential part (half step)
    psi = psi * np.exp(-1j * V * dt / (2 * hbar))
    
    return psi

fig = plt.figure(figsize=(16, 16))
plt.rcParams.update({'font.size': 10})
fig.suptitle('2D Quantum Tunneling Simulation', fontsize=18, y=0.98)

gs = fig.add_gridspec(5, 3, height_ratios=[6, 3, 3, 1.5, 1], width_ratios=[2, 2, 1], 
                      hspace=0.7, wspace=0.4, left=0.08, right=0.95, bottom=0.05, top=0.92)

explanation_text = """Quantum Tunneling Physics:
• Particles can tunnel through barriers even when E < V
• Higher E/V ratio increases transmission probability
• Wave packet splits into reflected and transmitted parts
• When E > V, most of the wave passes through"""

plt.figtext(0.03, 0.95, explanation_text, ha="left", va="top", fontsize=11, 
           bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8))

#Panel 1: Top view (2D color map) - Full wave function
ax_top = fig.add_subplot(gs[0, :])
ax_top.set_title('Top View: Full Wave Packet')
ax_top.set_xlabel('x')
ax_top.set_ylabel('y')

#barrier display
barrier_left = barrier_position - barrier_width/2
barrier_right = barrier_position + barrier_width/2

#top view display
probability = np.abs(psi)**2
img = ax_top.imshow(
    probability,
    extent=[x_min, x_max, y_min, y_max],
    origin='lower',
    cmap='viridis',
    vmin=0,
    vmax=np.max(probability) * 1.5
)
cbar = plt.colorbar(img, ax=ax_top, label='Probability Density', pad=0.01)
cbar.ax.tick_params(labelsize=9)

#barrier on top view (red rectangle)
barrier_patch = patches.Rectangle(
    (barrier_left, y_min),
    barrier_width,
    y_max - y_min,
    linewidth=2,
    edgecolor='r',
    facecolor='r',
    alpha=0.3
)
ax_top.add_patch(barrier_patch)

#Panel 2: Incident + Reflected Wave (Left side)
ax_left = fig.add_subplot(gs[1, 0:2])
ax_left.set_title('Incident + Reflected Wave (Left of Barrier)', pad=10)
ax_left.set_xlabel('x')
ax_left.set_ylabel('Probability Density')
ax_left.set_xlim(x_min, barrier_left)
ax_left.set_ylim(0, 0.8)

#Panel 3: Transmitted Wave (Right side)
ax_right = fig.add_subplot(gs[2, 0:2])
ax_right.set_title('Transmitted Wave (Right of Barrier)', pad=10)
ax_right.set_xlabel('x')
ax_right.set_ylabel('Probability Density')
ax_right.set_xlim(barrier_right, x_max)
ax_right.set_ylim(-0.001, 0.1)  #smaller scale that will auto-adjust

#Panel 4: Side view (1D slice along x-axis)
ax_side = fig.add_subplot(gs[1:3, 2])
ax_side.set_title('Side View: Wave Packet and Potential', pad=10)
ax_side.set_xlabel('x')
ax_side.set_ylabel('Probability / Potential')
ax_side.set_xlim(x_min, x_max)
ax_side.set_ylim(0, 2.5)

#side view with the central slice of the wave function (y=0)
central_slice = probability[Ny//2, :]
line_prob, = ax_side.plot(x, central_slice, 'b-', label='Probability Density')

#Add potential barrier to side view
potential_x = np.linspace(x_min, x_max, Nx)
potential_y = np.zeros_like(potential_x)
potential_y[(potential_x >= barrier_left) & (potential_x <= barrier_right)] = barrier_height
line_potential, = ax_side.plot(potential_x, potential_y, 'r-', label='Potential')

line_energy, = ax_side.plot(
    [x_min, x_max],
    [energy, energy],
    'g--',
    label='Energy'
)

ax_side.legend(loc='upper right', fontsize=8)

#Initialize the incident+reflected and transmitted wave displays
#Extract left and right regions
left_region_x = x[x < barrier_left]
right_region_x = x[x > barrier_right]

left_region_prob = central_slice[x < barrier_left]
right_region_prob = central_slice[x > barrier_right]

line_left, = ax_left.plot(left_region_x, left_region_prob, 'b-', label='Left Region')
line_right, = ax_right.plot(right_region_x, right_region_prob, 'g-', label='Right Region')

ax_trans = fig.add_subplot(gs[3, 0:3])
ax_trans.set_title('Transmission & Reflection Over Time', pad=10)
ax_trans.set_xlabel('Time')
ax_trans.set_ylabel('Probability (%)')
ax_trans.set_xlim(0, frames)
ax_trans.set_ylim(0, 100)

transmission_history = []
times = []
line_trans, = ax_trans.plot(times, transmission_history, 'g-', label='Transmission')
line_refl, = ax_trans.plot(times, transmission_history, 'r-', label='Reflection')

ax_trans.legend(loc='upper left', fontsize=9, framealpha=0.9)

ax_energy = fig.add_subplot(gs[4, 0:2])
energy_slider = Slider(
    ax=ax_energy,
    label='Particle Energy',
    valmin=0.1,
    valmax=4.0,
    valinit=energy,
    valstep=0.1
)

ax_barrier = fig.add_subplot(gs[4, 2])
barrier_slider = Slider(
    ax=ax_barrier,
    label='Barrier Height',
    valmin=0.1,
    valmax=5.0,
    valinit=barrier_height,
    valstep=0.1
)

def calculate_transmission(energy, barrier_height):
    """
    Calculate the theoretical transmission coefficient.
    
    This uses the quantum tunneling formula for a rectangular barrier:
    For E < V: T ≈ exp(-2κL) where κ = √(2m(V-E))/ħ
    For E > V: T depends on resonant conditions
    """
    if barrier_height <= 0:
        return 100.0  #No barrier
        
    if energy < barrier_height:
        #Tunneling case (E < V)
        k1 = np.sqrt(2 * m * energy) / hbar  #Wave number outside barrier
        k2 = np.sqrt(2 * m * (barrier_height - energy)) / hbar  #Imaginary wave number inside
        #Avoid division by zero
        try:
            #Full quantum tunneling formula for rectangular barrier
            T = 1 / (1 + (barrier_height**2 * np.sinh(k2 * barrier_width)**2) / 
                    (4 * energy * (barrier_height - energy)))
        except:
            #Simplified formula for thick barriers
            T = np.exp(-2 * k2 * barrier_width)
    else:
        #Above barrier case (E > V)
        k1 = np.sqrt(2 * m * energy) / hbar  # Wave number outside
        k2 = np.sqrt(2 * m * (energy - barrier_height)) / hbar  # Wave number inside
        try:
            #Formula for transmission when E > V (resonance effects included)
            T = 1 / (1 + (barrier_height**2 * np.sin(k2 * barrier_width)**2) / 
                    (4 * energy * (energy - barrier_height)))
        except:
            #High energy approximation
            T = 0.9  #High transmission for E >> V
    
    #Ensure physical result
    if np.isnan(T) or T < 0:
        T = 0.01
    elif T > 1:
        T = 1.0
        
    return 100 * T

#Pre-calculate frames with the split-step method
psi_frames = []
current_psi = psi.copy()
for i in range(frames):
    current_psi = evolve_wavefunction(current_psi, V, dt)
    psi_frames.append(current_psi.copy())

animation_running = True

def update_simulation(val=None):
    global psi_frames, V, energy, barrier_height, psi, barrier_patch
    global animation_running, transmission_history, times
    
    #Pause the animation
    if animation_running:
        animation_running = False
        ani.event_source.stop()
    
    energy = energy_slider.val
    barrier_height = barrier_slider.val
    
    #Update potential
    V = create_potential(barrier_height, barrier_width, barrier_position)
    
    #Update barrier visualization
    barrier_patch.set_alpha(min(0.8, barrier_height / 5.0))
    
    #Update energy line
    line_energy.set_ydata([energy, energy])
    
    #Update potential line in side view
    potential_y = np.zeros_like(potential_x)
    potential_y[(potential_x >= barrier_left) & (potential_x <= barrier_right)] = barrier_height
    line_potential.set_ydata(potential_y)
    
    #Adjust y-axis of side view if needed
    max_y = max(barrier_height * 1.2, energy * 1.2, 1.5)
    ax_side.set_ylim(0, max_y)
    
    #Update wave packet with new energy
    initial_momentum = np.sqrt(2 * m * energy)
    psi = create_wave_packet(initial_position, initial_momentum, sigma)
    
    #Recalculate frames with the split-step method
    psi_frames = []
    current_psi = psi.copy()
    for i in range(frames):
        current_psi = evolve_wavefunction(current_psi, V, dt)
        psi_frames.append(current_psi.copy())
    
    #Reset transmission history
    transmission_history = []
    times = []
    
    #Restart animation
    ani.frame_seq = ani.new_frame_seq()
    animation_running = True
    ani.event_source.start()

    fig.canvas.draw_idle()

energy_slider.on_changed(update_simulation)
barrier_slider.on_changed(update_simulation)

def animate(i):
    if not animation_running:
        return []
        
    #Update top view (full wave function)
    if i < len(psi_frames):
        current_psi = psi_frames[i]
        probability = np.abs(current_psi)**2
        img.set_array(probability)
        
        #Update side view (central slice)
        central_slice = probability[Ny//2, :]
        line_prob.set_ydata(central_slice)
        
        #Update left and right regions for separate incident/reflected and transmitted views
        left_region_prob = central_slice[x < barrier_left]
        right_region_prob = central_slice[x > barrier_right]
        
        #Make sure we're using the correct arrays for plotting
        line_left.set_ydata(left_region_prob)
        
        #Check if the shapes match for right region
        if len(right_region_x) == len(right_region_prob):
            line_right.set_ydata(right_region_prob)
            
            # Ensure the right plot shows data by scaling appropriately
            max_right = np.max(right_region_prob)
            if max_right > 0.001:  #Only adjust if there's actual data
                ax_right.set_ylim(0, max_right * 1.5)
            else:
                ax_right.set_ylim(0, 0.1)  #Default scale when no wave has passed through
        
        #Auto-scale the left and right plots to show detail
        max_left = np.max(left_region_prob)
        if max_left > 0.001:
            ax_left.set_ylim(0, max(0.8, max_left * 1.2))
        
        #Calculate transmission/reflection
        left_region = (X < (barrier_position - barrier_width/2))
        right_region = (X > (barrier_position + barrier_width/2))
        
        total_prob = np.sum(probability) * dx * dy
        right_prob = np.sum(probability * right_region) * dx * dy
        left_prob = np.sum(probability * left_region) * dx * dy
        
        #Calculate transmission/reflection percentages
        trans_percent = 100 * right_prob / total_prob
        refl_percent = 100 * left_prob / total_prob
        
        #Add to history
        transmission_history.append(trans_percent)
        times.append(i)
        
        #Update transmission plot
        line_trans.set_data(times, transmission_history)
        line_refl.set_data(times, [100 - t for t in transmission_history])
        
        #Update title with current E/V ratio
        ratio = energy / barrier_height if barrier_height > 0 else "∞"
        ax_top.set_title(f'Top View: Full Wave Packet (E/V Ratio = {ratio:.2f})', pad=10)
        
        #Show theoretical transmission
        theory_trans = calculate_transmission(energy, barrier_height)
        
        #Update titles for the separated wave function views
        ax_left.set_title(f'Incident + Reflected Wave (Left of Barrier)')
        ax_right.set_title(f'Transmitted Wave (Right of Barrier) - Trans: {trans_percent:.1f}%')
        
        #Update barrier color based on E/V ratio
        if energy < barrier_height:
            #Tunneling (red barrier)
            barrier_patch.set_facecolor('r')
        else:
            #Above barrier (yellow barrier)
            barrier_patch.set_facecolor('y')
    
    return [img, line_prob, line_potential, line_energy, line_trans, line_refl, 
            barrier_patch, line_left, line_right]

#Create animation with smoother framerate
ani = FuncAnimation(
    fig,
    animate,
    frames=frames,
    interval=50,  #Shorter interval for smoother animation
    blit=True,
    cache_frame_data=False
)

#Don't use tight_layout or subplots_adjust as the gridspec has explicit parameters
plt.show()