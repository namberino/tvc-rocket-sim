import numpy as np

def three_dof_body_axes(
    Fx, Fz, My, 
    u0=100.0, w0=0.0, theta0=0.0, q0=0.0, alpha0=0.0, pos0=(0.0, 0.0),
    mass=0.543, mass_e=0.432, mass_f=0.543, Iyy_e=0.028, Iyy_f=0.048,
    g=9.81, include_inertial_acceleration=False
):
    """
    Simulates the 3DoF (Body Axes) dynamics of a rigid body
    
    Input parameters:
    - Fx: Force in the body x-direction (N)
    - Fz: Force in the body z-direction (N)
    - My: Moment about the body y-axis (Nm)
    - u0: Initial velocity in the body x-direction (m/s)
    - w0: Initial velocity in the body z-direction (m/s)
    - theta0: Initial pitch angle (rad)
    - q0: Initial pitch angular rate (rad/s)
    - alpha0: Initial angle of attack (rad)
    - pos0: Initial position in the flat Earth frame (m, m)
    - mass: Initial mass (kg)
    - mass_e: Empty mass (kg)
    - mass_f: Full mass (kg)
    - Iyy_e: Empty moment of inertia about the y-axis (kg·m²)
    - Iyy_f: Full moment of inertia about the y-axis (kg·m²)
    - g: Acceleration due to gravity (m/s²)
    - include_inertial_acceleration: Whether to include inertial acceleration output
    
    Output return values:
    - theta: Pitch attitude (rad)
    - q: Pitch angular rate (rad/s)
    - dqdt: Pitch angular acceleration (rad/s²)
    - pos: Position in the flat Earth frame (m, m)
    - velocity: Velocity in the body frame (m/s, m/s)
    - acceleration: Acceleration in the body frame (m/s², m/s²)
    - inertial_acceleration: Acceleration in the inertial frame (m/s², m/s²) [if include_inertial_acceleration is True]
    """
    
    # calculate pitch angular acceleration
    Iyy = mass * (Iyy_f - Iyy_e) / (mass_f - mass_e) + Iyy_e
    dqdt = My / Iyy
    
    # update pitch angular rate
    q = q0 + dqdt
    
    # update pitch angle
    theta = theta0 + q
    
    # calculating linear accelerations
    Ax = Fx / mass
    Az = Fz / mass
    
    # update velocities
    u = u0 + Ax
    w = w0 + Az
    
    # update position
    x, z = pos0
    x += u * np.cos(theta) - w * np.sin(theta)
    z += u * np.sin(theta) + w * np.cos(theta)
    
    pos = (x, z)
    velocity = (u, w)
    acceleration = (Ax, Az)
    
    if include_inertial_acceleration:
        A_xe = Ax - g * np.sin(theta)
        A_ze = Az - g * np.cos(theta)
        inertial_acceleration = (A_xe, A_ze)

        return theta, q, dqdt, pos, velocity, acceleration, inertial_acceleration
    
    return theta, q, dqdt, pos, velocity, acceleration


def step_function(time, step_time=1.0, initial_value=0.0, final_value=1.0, sample_time=0.1, delay=0.0):
    """
    Generates a step signal

    Input parameters:
    - time: The current time (s)
    - step_time: The time at which the step occurs (s)
    - initial_value: The value of the output before the step time (float)
    - final_value: The value of the output after the step time (float)
    - sample_time: The time interval between samples (s)
    - delay: The delay before the step occurs (s)

    Output return values:
    - The value of the step signal at the given time (float)
    """
    effective_step_time = step_time + delay
    if time < effective_step_time:
        return initial_value
    else:
        return final_value


time_values = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0] # time points
step_time = 0.0
initial_value = 0.0
final_value = 15.0 # 15N of thrust
sample_time = 0.1
delay = 2.0

# generate step signal for each time point
step_signal = [step_function(t, step_time, initial_value, final_value, sample_time, delay) for t in time_values]
print(step_signal)



result = three_dof_body_axes(Fx=100, Fz=50, My=10)
print(result)

