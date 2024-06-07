import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def three_dof_body_axes(Fx, Fz, My, u0=0.0, w0=0.0, theta0=0.0, q0=0.0, pos0=[0.0, 0.0], mass=0, inertia=0.0, g=9.81, dt=0.01, duration=10):
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

    # time integration using Euler's method
    for t in np.arange(0, duration + dt, dt):  # start at dt and end at 'duration'
        # calculate accelerations
        ax = Fx[int(t/dt)] / mass
        az = Fz[int(t/dt)] / mass - g
        
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
        
        # stop if the rocket returns to ground level
        if pos[1] <= 0 and t > 2:  # allow some time for launch
            break
    
    return {
        'theta' : np.array(theta_list),
        'q' : np.array(q_list),
        'dqdt' : np.array(dqdt_list),
        'pos' : np.array(pos_list),
        'velocity' : np.array(velocity_list),
        'acceleration' : np.array(acceleration_list)
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
mass = 0.543 # kg
inertia = 0.048 # kg*m^2
g = 9.81 # m/s^2
peak_thrust = 15 # N
thrust_duration = 4 # s
simulation_duration = 15 # s
dt = 0.01 # time step
moment_arm = 0.28 # meters
gimbal_angle = 0.0001 # radian

# initial conditions
u0 = 0.0 # initial velocity in x (body axis)
w0 = 0.0 # initial velocity in z (body axis)
theta0 = 0.0 # initial pitch angle
q0 = 0.0 # initial pitch rate
pos0 = [0.0, 0.0] # initial position [x, z]

# generate thrust profile
# thrust_profile = generate_thrust_profile(simulation_duration, thrust_duration, peak_thrust, dt)
with open('f15_thrust_extended.npy', 'rb') as f:
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
results = three_dof_body_axes(Fx, Fz, My, u0, w0, theta0, q0, pos0, mass, inertia, g, dt, simulation_duration)

time = np.arange(0, len(results['pos']) * dt, dt)
theta = results['theta']
q = results['q']
pos = results['pos']
velocity = results['velocity']
acceleration = results['acceleration']

print(acceleration[:, 1])

fig, ax = plt.subplots(figsize=(10, 6)) # figure and axis

line, = ax.plot([], [], 'b-', label='Rocket Trajectory')
launch_point, = ax.plot([], [], 'go', label='Launch Point')
impact_point, = ax.plot([], [], 'ro', label='Impact Point')
ground_line = ax.axhline(0, color='black', linestyle='--', label='Ground')

# labels
ax.set_xlabel('X Position (m)')
ax.set_ylabel('Z Position (m)')
ax.set_title('Rocket Trajectory Animation')
ax.legend()

# axis limits
ax.set_xlim(np.min(pos[:, 0]) - 1, np.max(pos[:, 0]) + 1)
ax.set_ylim(0, np.max(pos[:, 1]) + 1)

# text elements
theta_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
q_text = ax.text(0.02, 0.90, '', transform=ax.transAxes)
pos_text = ax.text(0.02, 0.85, '', transform=ax.transAxes)
vel_text = ax.text(0.02, 0.80, '', transform=ax.transAxes)
acc_text = ax.text(0.02, 0.75, '', transform=ax.transAxes)

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
    return line, launch_point, impact_point, theta_text, q_text, pos_text, vel_text, acc_text

# update plot for each frame
def update(frame):
    line.set_data(pos[:frame, 0], pos[:frame, 1])
    launch_point.set_data(pos[0, 0], pos[0, 1])
    impact_point.set_data(pos[frame, 0], pos[frame, 1])
    
    theta_text.set_text(f'Theta: {theta[frame]:.2f} rad')
    q_text.set_text(f'Q: {q[frame]:.2f} rad/s')
    pos_text.set_text(f'Position: [{pos[frame, 0]:.2f}, {pos[frame, 1]:.2f}] m')
    vel_text.set_text(f'Velocity: [{velocity[frame, 0]:.2f}, {velocity[frame, 1]:.2f}] m/s')
    acc_text.set_text(f'Acceleration: [{acceleration[frame, 0]:.2f}, {acceleration[frame, 1]:.2f}] m/s²')

    return line, launch_point, impact_point, theta_text, q_text, pos_text, vel_text, acc_text

animation = FuncAnimation(fig, update, frames=len(pos), init_func=init, blit=True, interval=dt)
plt.show()
