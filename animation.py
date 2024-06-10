import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def calc_air_density(altitude, rho0=1.225, scale_height=8500):
    # calculate air density as a function of altitude
    return rho0 * np.exp(-altitude / scale_height)

def three_dof_body_axes(Fx, Fz, My, 
                        u0=0.0, w0=0.0, theta0=0.0, q0=0.0, pos0=[0.0, 0.0], 
                        mass=0, inertia=0.0, 
                        Cd=0.75, A=0.01, rho0=1.225, scale_height=8500,
                        g=9.81, dt=0.01, duration=10):
    # ensure pos0 is a float array
    pos = np.array(pos0, dtype=float)
    
    # initial conditions
    u = u0
    w = w0
    theta = theta0
    q = q0
    vel = np.array([u, w])
    
    # initial acceleration
    ax = Fx[0] / mass
    az = Fz[0] / mass - g
    
    # history lists to store values
    theta_list = [theta]
    q_list = [q]
    dqdt_list = [0]
    pos_list = [pos.copy()]
    velocity_list = [vel.copy()]
    acceleration_list = [np.array([ax, az])]
    density_list = [calc_air_density(pos[1], rho0, scale_height)]

    # time integration using Euler's method
    for t in np.arange(0, duration + dt, dt):  # start at 0 and end at 'duration'
        # calculate air drag force
        v_rocket = np.linalg.norm(vel)
        current_rho = calc_air_density(pos[1], rho0, scale_height)
        Fd = 1/2 * current_rho * v_rocket**2 * Cd * A
        Fd_x = Fd * (u / v_rocket) if v_rocket != 0 else 0
        Fd_z = Fd * (w / v_rocket) if v_rocket != 0 else 0

        # calculate accelerations
        # ax = Fx[int(t/dt)] / mass
        # az = Fz[int(t/dt)] / mass - g
        ax = (Fx[int(t/dt)] - Fd_x) / mass
        az = (Fz[int(t/dt)] - Fd_z) / mass - g
        
        # calculate angular acceleration
        dqdt = My[int(t/dt)] / inertia
        
        # update velocities
        u += ax * dt
        w += az * dt
        q += dqdt * dt
        
        # update positions
        pos += vel * dt
        vel = np.array([u, w])
        
        # update angle
        theta += q * dt
        
        # store data in list
        theta_list.append(theta)
        q_list.append(q)
        dqdt_list.append(dqdt)
        pos_list.append(pos.copy())
        velocity_list.append(vel.copy())
        acceleration_list.append(np.array([ax, az]))
        density_list.append(current_rho)
        
        # stop if the rocket returns to ground level
        if pos[1] <= 0 and t > 2:  # allow some time for launch
            break
    
    return {
        'theta' : np.array(theta_list),
        'q' : np.array(q_list),
        'dqdt' : np.array(dqdt_list),
        'pos' : np.array(pos_list),
        'velocity' : np.array(velocity_list),
        'acceleration' : np.array(acceleration_list),
        'air_density' : np.array(density_list)
    }

def generate_thrust_profile(duration, thrust_duration, peak_thrust, dt=0.01):
    thrust_profile = []
    for t in np.arange(0, duration + dt, dt):
        if t < 0.1 * thrust_duration:
            # ignition and rapid rise (modeled as quadratic rise)
            thrust = peak_thrust * (10 * t / thrust_duration)**2
        elif t < 0.3 * thrust_duration:
            # peak thrust
            thrust = peak_thrust
        elif t < thrust_duration:
            # decay phase (modeled as linear decay)
            thrust = peak_thrust * (1 - (t - 0.3 * thrust_duration) / (0.7 * thrust_duration))
        else:
            # burnout
            thrust = 0
        thrust_profile.append(thrust)
    return np.array(thrust_profile)

# parameters
Cd = 0.75 # air drag coefficient
A = 0.009 # reference area (m^2)
rho0 = 1.225 # air density (kg/m^3)
scale_height = 8500 # m (altitude where atmospheric pressure decreases by a factor of e)
mass = 0.543 # kg
inertia = 0.048 # kg*m^2
g = 9.81 # m/s^2
peak_thrust = 15 # N
thrust_duration = 4 # s
simulation_duration = 15 # s
dt = 0.01 # time step
moment_arm = 0.28 # meters
gimbal_angle = 0.001 # radian

# initial conditions
u0 = 0.0 # initial velocity in x (body axis)
w0 = 0.0 # initial velocity in z (body axis)
theta0 = 0.0 # initial pitch angle
q0 = 0.0 # initial pitch rate
pos0 = [0.0, 0.0] # initial position [x, z]

# generate thrust profile
# thrust_profile = generate_thrust_profile(simulation_duration, thrust_duration, peak_thrust, dt)
with open('profiles/f15_thrust_extended.npy', 'rb') as f:
    thrust_profile = np.load(f)
print(max(thrust_profile))

# initialize forces and moments
Fx = np.sin(gimbal_angle) * thrust_profile # horizontal thrust
Fz = np.cos(gimbal_angle) * thrust_profile # vertical thrust
My = Fx * moment_arm # pitching moment (torque)

print(Fx)
print(Fz)
print(My)

# simulation
results = three_dof_body_axes(Fx, Fz, My, 
                              u0, w0, theta0, q0, pos0, 
                              mass, inertia, 
                              Cd, A, rho0, scale_height, 
                              g, dt, simulation_duration)

time = np.arange(0, len(results['pos']) * dt, dt)
theta = results['theta']
q = results['q']
pos = results['pos']
velocity = results['velocity']
acceleration = results['acceleration']
air_density = results['air_density']

print(acceleration[:, 1])

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12)) # two subplots

# Trajectory plot
line, = ax1.plot([], [], 'b-', label='Rocket Trajectory')
launch_point, = ax1.plot([], [], 'go', label='Launch Point')
impact_point, = ax1.plot([], [], 'ro', label='Impact Point')
ground_line = ax1.axhline(0, color='black', linestyle='--', label='Ground')

# labels
ax1.set_xlabel('X Position (m)')
ax1.set_ylabel('Z Position (m)')
ax1.set_title('Rocket Trajectory Animation')
ax1.legend()

# axis limits
ax1.set_xlim(np.min(pos[:, 0]) - 1, np.max(pos[:, 0]) + 1)
ax1.set_ylim(0, np.max(pos[:, 1]) + 1)

# text elements for trajectory
theta_text = ax1.text(0.02, 0.95, '', transform=ax1.transAxes)
q_text = ax1.text(0.02, 0.90, '', transform=ax1.transAxes)
pos_text = ax1.text(0.02, 0.85, '', transform=ax1.transAxes)
vel_text = ax1.text(0.02, 0.80, '', transform=ax1.transAxes)
acc_text = ax1.text(0.02, 0.75, '', transform=ax1.transAxes)
density_text = ax1.text(0.02, 0.70, '', transform=ax1.transAxes)

# Thrust profile plot
thrust_line, = ax2.plot([], [], 'r-', label='Thrust Profile')
current_thrust, = ax2.plot([], [], 'ro', label='Current Thrust')

# labels
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Thrust (N)')
ax2.set_title('Thrust Profile Animation')
ax2.legend()

# axis limits
ax2.set_xlim(0, simulation_duration)
ax2.set_ylim(0, np.max(thrust_profile) + 1)

# text element for thrust
thrust_text = ax2.text(0.05, 0.90, '', transform=ax2.transAxes)
max_thrust_value = 0
max_thrust_text = ax2.text(0.05, 0.85, '', transform=ax2.transAxes)

# initialize points for the 1st frame
def init():
    line.set_data([], [])
    launch_point.set_data([], [])
    impact_point.set_data([], [])
    theta_text.set_text('')
    q_text.set_text('')
    pos_text.set_text('')
    vel_text.set_text('')
    acc_text.set_text('')
    density_text.set_text('')
    
    thrust_line.set_data([], [])
    current_thrust.set_data([], [])
    thrust_text.set_text('')
    max_thrust_text.set_text('')
    
    return line, launch_point, impact_point, theta_text, q_text, pos_text, vel_text, acc_text, density_text, thrust_line, current_thrust, thrust_text, max_thrust_text

# update plot for each frame
def update(frame):
    global max_thrust_value

    line.set_data(pos[:frame, 0], pos[:frame, 1])
    launch_point.set_data(pos[0, 0], pos[0, 1])
    impact_point.set_data(pos[frame, 0], pos[frame, 1])
    
    theta_text.set_text(f'Theta: {theta[frame]:.2f} rad')
    q_text.set_text(f'Q: {q[frame]:.2f} rad/s')
    pos_text.set_text(f'Position: [{pos[frame, 0]:.2f}, {pos[frame, 1]:.2f}] m')
    vel_text.set_text(f'Velocity: [{velocity[frame, 0]:.2f}, {velocity[frame, 1]:.2f}] m/s')
    acc_text.set_text(f'Acceleration: [{acceleration[frame, 0]:.2f}, {acceleration[frame, 1]:.2f}] m/s²')
    density_text.set_text(f'Air density: {air_density[frame]:.3f} kg/m³')

    thrust_line.set_data(time[:frame], thrust_profile[:frame])
    current_thrust.set_data(time[frame], thrust_profile[frame])
    thrust_text.set_text(f'Current Thrust: {thrust_profile[frame]:.2f} N')

    if thrust_profile[frame] > max_thrust_value:
        max_thrust_value = thrust_profile[frame]

    max_thrust_text.set_text(f'Max Thrust: {max_thrust_value:.2f} N')

    return line, launch_point, impact_point, theta_text, q_text, pos_text, vel_text, acc_text, density_text, thrust_line, current_thrust, thrust_text, max_thrust_text

animation = FuncAnimation(fig, update, frames=len(pos), init_func=init, blit=True, interval=dt)
plt.show()
