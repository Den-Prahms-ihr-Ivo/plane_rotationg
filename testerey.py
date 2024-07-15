import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

plane_width = 3
plane_length = 7
plane_space = 0.5
plane_depth = 0.7


def create_paper_plane(
    plane_width=3,
    plane_length=7,
    plane_space=0.5,
    plane_depth=0.7,
    marble_x=0,
    marble_y=3,
    marble_z=0.5,
):
    marble = np.array([marble_x, marble_y, marble_z])

    sides = np.zeros((4, 3, 2, 2))

    sides[0][0] = np.array([[-plane_width, 0], [-plane_space, 0]])
    sides[0][1] = np.array([[0, plane_length], [0, plane_length]])
    sides[0][2] = np.array([[0, 0], [0, 0]])
    sides[1][0] = np.array([[plane_width, 0], [plane_space, 0]])
    sides[1][1] = np.array([[0, plane_length], [0, plane_length]])
    sides[1][2] = np.array([[0, 0], [0, 0]])

    sides[2][0] = np.array([[-plane_space, 0], [0, 0]])
    sides[2][1] = np.array([[0, plane_length], [0, plane_length]])
    sides[2][2] = np.array([[0, 0], [-plane_depth, 0]])

    sides[3][0] = np.array([[plane_space, 0], [0, 0]])
    sides[3][1] = np.array([[0, plane_length], [0, plane_length]])
    sides[3][2] = np.array([[0, 0], [-plane_depth, 0]])
    return sides, marble


marble = None
# create the figure
fig = plt.figure()

# add axes
ax = fig.add_subplot(111, projection="3d")
ax.set_xlim([-5, 5])
ax.set_ylim([-2, 8])
ax.set_zlim([-2, 2])
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")


sides, marble = create_paper_plane()
# plot the plane
ax.plot_surface(
    sides[0][0],
    sides[0][1],
    sides[0][2],
    alpha=0.5,
    color=[0, 0.5, 0.9, 0.5],
    linewidth=0.9,
    edgecolors=[0, 0, 0, 0.9],
)
ax.plot_surface(
    sides[1][0],
    sides[1][1],
    sides[1][2],
    alpha=0.5,
    color=[0, 0.5, 0.9, 0.5],
    linewidth=0.9,
    edgecolors=[0, 0, 0, 0.9],
)
ax.plot_surface(
    sides[2][0],
    sides[2][1],
    sides[2][2],
    alpha=0.5,
    color=[0, 0.5, 0.9, 0.5],
    linewidth=0.9,
    edgecolors=[0, 0, 0, 0.9],
)
ax.plot_surface(
    sides[3][0],
    sides[3][1],
    sides[3][2],
    alpha=0.5,
    color=[0, 0.5, 0.9, 0.5],
    linewidth=0.9,
    edgecolors=[0, 0, 0, 0.9],
)

if marble is not None:
    ax.plot3D(
        xs=marble[0],
        ys=marble[1],
        zs=marble[2],
        zorder=100,
        color=[0.9, 0, 0, 1],
        marker="o",
        linewidth=2,
        markersize=9,
    )


plt.show()
