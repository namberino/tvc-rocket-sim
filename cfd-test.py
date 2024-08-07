import numpy as np
import matplotlib.pyplot as plt

def make_rocket(Nx, Ny):
    # Ensure the rocket shape fits within Nx and Ny
    rocket_length = Nx // 2
    rocket_height = Ny // 2

    # Create a blank canvas
    cylinder = np.full((Ny, Nx), False)

    # Center y-coordinate
    center_y = Ny // 2

    # Define the radius (height/2) of the cone base
    cone_radius = rocket_height // 2

    # Define the end of the cone base
    cone_base_end = cone_radius * 2

    # Calculate the offset to center the rocket along the x-axis
    x_offset = (Nx - rocket_length - 150) // 2

    # Fill the cone shape pointing to the left
    for y in range(center_y - cone_radius, center_y + cone_radius):
        for x in range(cone_base_end):
            if abs(center_y - y) <= x * (cone_radius / cone_base_end):
                cylinder[y][x + x_offset] = True

    # Fill the rectangular part and the flat right end
    for y in range(center_y - cone_radius, center_y + cone_radius):
        for x in range(cone_base_end, rocket_length):
            cylinder[y][x + x_offset] = True

    # Define the radius for the rounded corners
    corner_radius = 10  # Smaller radius for a smaller rocket

    # Smooth and round the bottom-right corner
    for y in range(center_y, center_y + corner_radius):
        for x in range(rocket_length - corner_radius, rocket_length):
            if (x - (rocket_length - corner_radius))**2 + (y - (center_y + corner_radius))**2 > corner_radius**2:
                cylinder[y][x + x_offset] = False

    # Smooth and round the top-right corner
    for y in range(center_y - corner_radius, center_y):
        for x in range(rocket_length - corner_radius, rocket_length):
            if (x - (rocket_length - corner_radius))**2 + (y - (center_y - corner_radius))**2 > corner_radius**2:
                cylinder[y][x + x_offset] = False

    # Extend the space between the two rounded corners further to the right
    for y in range(center_y - corner_radius, center_y + corner_radius):
        for x in range(rocket_length - corner_radius, rocket_length):
            cylinder[y][x + x_offset] = True

    return cylinder

plot_iteration = 5 # plot every n iteration

def distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

# number of lattice in each directions
Nx = 400
Ny = 100

tau = 0.53 # kinematic viscosity
Nt = 30000 # iterations

# lattice speeds and weights
NL = 9 # number of lattices

# discrete velocity of each nodes
cxs = np.array([0, 0, 1, 1, 1, 0, -1, -1, -1])
cys = np.array([0, 1, 1, 0, -1, -1, -1, 0, 1])
weights = np.array([4/9, 1/9, 1/36, 1/9, 1/36, 1/9, 1/36, 1/9, 1/36])

# initial condition (velocity of each cells' nodes)
F = np.ones((Ny, Nx, NL)) + 0.01 * np.random.randn(Ny, Nx, NL) # the randomness introduces inconsistency
F[:, :, 3] = 2.3 # at every lattice, in each of their 3rd node

# an bool array that maps to every cells (false means empty space)
cylinder = np.full((Ny, Nx), False)

side_length = 12
half_side = side_length // 2

# set cylinder boundary (radius = 12)
for y in range(0, Ny):
    for x in range(0, Nx):
        # 1/4 of the way into the X axis, 1/2 the way into the Y axis
        if distance(Nx // 4, Ny // 2, x, y) < 12:
            cylinder[y][x] = True
        # center_x = Nx // 4
        # center_y = Ny // 2
        # if (center_x - half_side <= x <= center_x + half_side) and (center_y - half_side <= y <= center_y + half_side):
        #     cylinder[y][x] = True

cylinder = make_rocket(Nx, Ny)

# time stepping
for t in range(Nt):
    if t + 1 % 100 == 0:
        print(f'Iteration {t + 1}')

    # at the right wall
    F[:, -1, [6, 7, 8]] = F[:, -2, [6, 7, 8]] # set to the value of the nodes next to it so they cancel each other out

    F[:, 0, [2, 3, 4]] = F[:, 1, [2, 3, 4]] # same thing for the left wall

    # stream (move) the nodal velocity to nodes of the neighboring lattices
    for i, cx, cy in zip(range(NL), cxs, cys):
        # go through every node and roll it in the direction of its corresponding discrete vel
        F[:, :, i] = np.roll(F[:, :, i], cx, axis=1)
        F[:, :, i] = np.roll(F[:, :, i], cy, axis=0)

    # invert velocity of lattices in the boundary (bounce off)
    boundaryF = F[cylinder, :] # get points where vel is inside the cylinder
    boundaryF = boundaryF[:, [0, 5, 6, 7, 8, 1, 2, 3, 4]] # invert vel when hits the cylinder (collision)

    # fluid variables
    rho = np.sum(F, 2) # density (sum of the last axis - the axis with all the velocity)
    # momentum
    ux = np.sum(F * cxs, 2) / rho
    uy = np.sum(F * cys, 2) / rho

    # apply boundary
    F[cylinder, :] = boundaryF
    ux[cylinder] = 0 # no movement in the boundary (in the solid object)
    uy[cylinder] = 0 # no movement in the boundary (in the solid object)

    # collision
    Feq = np.zeros(F.shape)
    for i, cx, cy, w in zip(range(NL), cxs, cys, weights):
        Feq[:, :, i] = rho * w * (1 + 3 * (cx * ux + cy * uy) + 9 * (cx * ux + cy * uy)**2 / 2 - 3 * (ux**2 + uy**2) / 2)
    
    F = F + -(1 / tau) * (F - Feq)

    if t % plot_iteration == 0:
        # plot curl equation
        # dfydx = ux[2:, 1:-1] - ux[0:-2, 1:-1]
        # dfxdy = uy[1:-1, 2:] - uy[1:-1, 0:-2]
        # curl = dfydx - dfxdy

        plt.imshow(np.sqrt(ux**2 + uy**2), cmap='rainbow') # visualize magnitude of velocity
        # plt.imshow(curl, cmap='bwr') # plot the curl
        plt.pause(0.000000000000000000000000000000001)
        plt.cla()
