import numpy as np

def three_dof_body_axes(
    Fx, Fz, My, 
    u0=100.0, w0=0.0, theta0=0.0, q0=0.0, alpha0=0.0, pos0=(0.0, 0.0),
    mass=0.543, mass_e=0.432, mass_f=0.543, Iyy_e=0.028, Iyy_f=0.048,
    g=9.81, include_inertial_acceleration=False
):
    """
    Simulates the 3DoF (Body Axes) dynamics of a rigid body.
    
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

# Example of usage
result = three_dof_body_axes(Fx=100, Fz=50, My=10)
print(result)

