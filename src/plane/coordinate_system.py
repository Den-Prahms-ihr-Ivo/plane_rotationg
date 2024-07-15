from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d

# from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from src.helper.types import Array, Point

from typing import Optional, List

import numpy as np


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))

        return np.min(zs)


def vector_from_points(points: List[Point]):
    """
    Turns two points in space into a vector representation, many matplotlib functions want to see.
    e.g.:

      x,y,z   x,y,z
    [[1,2,3],[4,5,6]] ==> x: [[1,4]]
                          y: [[2,5]]
                          z: [[3,6]]

    To convert two points into a vector representation, you can call this function like this:
    np.array([[0,0,0], [1,0,0]])

    If you want to convert multiple points, for examples a standard coordinate System:

     ------- X -------  ------- Y -------  ------- Z -------
    [[[0,0,0],[1,0,0]], [[0,0,0],[0,1,0]], [[0,0,0],[0,0,1]]] ==> x: [[0,1], [0,0], [0,0]]
                                                                  y: [[0,0], [1,0], [0,0]]
                                                                  z: [[0,0], [0,0], [1,0]]

    """
    _x = []
    _y = []
    _z = []

    # Single Point
    if len(points.shape) == 2:
        _x = [points[0][0], points[1][0]]
        _y = [points[0][1], points[1][1]]
        _z = [points[0][2], points[1][2]]

    # Multiple Points
    else:
        for p in points:

            _x.append([p[0][0], p[1][0]])
            _y.append([p[0][1], p[1][1]])
            _z.append([p[0][2], p[1][2]])

    return np.array(_x), np.array(_y), np.array(_z)


def draw_line(ax, point_A, point_B, color="r", lw=2):
    x, y, z = vector_from_points(np.array([point_A, point_B]))

    arrow_prop_dict = dict(
        mutation_scale=20, lw=lw, zorder=-2, arrowstyle="-", shrinkA=0, shrinkB=0
    )

    a = Arrow3D(x, y, z, color=color, **arrow_prop_dict)
    ax.add_artist(a)


class Axis:
    """
    Arrows for the coordinate system:
    TODO: Wäre eine Klasse "Pfeil" oder "Axis" schön,
    die aus zwei Punkten im Raum einen Vector erstellt. Dazu wäre die Funktion Transform und Translate Sinnvoll.
    """

    # Points making up an Axis
    points: np.array((2, 3))

    # Vector Representation of Points making up the axis
    x_vector: np.array((3, 2))
    y_vector: np.array((3, 2))
    z_vector: np.array((3, 2))

    # npt.NDArray[np.float64]
    def __init__(
        self,
        points: Optional[Array["2,3", int]] = None,
        point_A: Optional[Array[3, int]] = None,
        point_B: Optional[Array[3, int]] = None,
    ):
        if points is not None:
            self.points = np.array(points)
        elif point_A is not None and point_B is not None:
            self.points = np.array([point_A, point_B])
        else:
            raise ValueError("Please provide valid arguments")

        assert self.points.shape == (2, 3)

        self.x_vector, self.y_vector, self.z_vector = vector_from_points(self.points)

    def matrix_rotate(self, rotation_matrix):
        self.points = np.array([np.matmul(rotation_matrix, x) for x in self.points])

        # Update Vectors
        self.x_vector, self.y_vector, self.z_vector = vector_from_points(self.points)
        return self

    def translate(self, x=0, y=0, z=0):
        # Translation matrix
        mat = [
            [1, 0, 0, x],
            [0, 1, 0, y],
            [0, 0, 1, z],
            [0, 0, 0, 1],
        ]

        # temporary extension of the current points to be able to multiply it. e.g
        # [[0,0,0],[1,0,0]] -> [[0,0,0,1],[1,0,0,1]]
        nina = np.append(self.points, np.ones((len(self.points), 1)), axis=1)

        # multiply Points by translation matrix
        self.points = np.array([np.matmul(mat, a)[:-1] for a in nina])

        # Update Vectors
        self.x_vector, self.y_vector, self.z_vector = vector_from_points(self.points)
        return self

    def matrix_translate(self, mat):
        # temporary extension of the current points to be able to multiply it. e.g
        # [[0,0,0],[1,0,0]] -> [[0,0,0,1],[1,0,0,1]]
        nina = np.append(self.points, np.ones((len(self.points), 1)), axis=1)

        # multiply Points by translation matrix
        self.points = np.array([np.matmul(mat, a)[:-1] for a in nina])

        # Update Vectors
        self.x_vector, self.y_vector, self.z_vector = vector_from_points(self.points)
        return self

    def translate_on_axis(self):
        raise NotImplementedError

    def __str__(self):
        return f"Made up by the points: [{self.points[0][0]},{self.points[0][1]},{self.points[0][2]}] -> [{self.points[1][0]},{self.points[1][1]},{self.points[1][2]}]"

    def __repr__(self):
        return self.__str__()


class CoordinateSystem:
    """
    Repräsentation eines Koordinatensystems bestehend aus 3 Achsen.
    Mit der Möglichkeit zur Transformation und Translation mit Matrizen

    Usage:
    Entweder gibt man die Punkte für die Achsen in der Form eines Array ein
                                     ------- X -------  ------- Y -------  ------- Z -------
    cs = CoordinateSystem(points=[   [[0,0,0],[1,0,0]], [[0,0,0],[0,1,0]], [[0,0,0],[0,0,1]]   ])

    oder man gibt die Achsen explizit an über X_axis = axis
    """

    X_axis: Axis
    Y_axis: Axis
    Z_axis: Axis

    def __init__(
        self,
        points: Optional[Array["3,2,3", int]] = None,
        X_axis: Optional[Axis] = None,
        Y_axis: Optional[Axis] = None,
        Z_axis: Optional[Axis] = None,
    ):
        if points is not None:
            # self.
            points = np.array(points)
            assert points.shape == (3, 2, 3)

            self.X_axis = Axis(points[0])
            self.Y_axis = Axis(points[1])
            self.Z_axis = Axis(points[2])

        elif X_axis is not None and Y_axis is not None and Z_axis:
            self.X_axis = X_axis
            self.Y_axis = Y_axis
            self.Z_axis = Z_axis
        else:
            raise ValueError("Please provide valid arguments")

    def plot_coordinates(
        self,
        ax: Axes3D,
        lw=2,
        label_offset=0.01,
        colors=["r", "g", "b"],
        show_labels=True,
        label_font_size=9,
    ):
        arrow_prop_dict = dict(
            mutation_scale=20, lw=lw, zorder=-2, arrowstyle="->", shrinkA=0, shrinkB=0
        )

        a = Arrow3D(
            self.X_axis.x_vector,
            self.X_axis.y_vector,
            self.X_axis.z_vector,
            color=colors[0],
            **arrow_prop_dict,
        )
        ax.add_artist(a)
        a = Arrow3D(
            self.Y_axis.x_vector,
            self.Y_axis.y_vector,
            self.Y_axis.z_vector,
            color=colors[1],
            **arrow_prop_dict,
        )
        ax.add_artist(a)
        a = Arrow3D(
            self.Z_axis.x_vector,
            self.Z_axis.y_vector,
            self.Z_axis.z_vector,
            color=colors[2],
            **arrow_prop_dict,
        )
        ax.add_artist(a)

        if show_labels:
            ax.text(
                self.X_axis.x_vector[1] + label_offset,
                self.X_axis.y_vector[1] + label_offset,
                self.X_axis.z_vector[1] + label_offset,
                r"$x$",
                fontsize=label_font_size,
            )
            ax.text(
                self.Y_axis.x_vector[1] + label_offset,
                self.Y_axis.y_vector[1] + label_offset,
                self.Y_axis.z_vector[1] + label_offset,
                r"$y$",
                fontsize=label_font_size,
            )
            ax.text(
                self.Z_axis.x_vector[1] + label_offset,
                self.Z_axis.y_vector[1] + label_offset,
                self.Z_axis.z_vector[1] + label_offset,
                r"$z$",
                fontsize=label_font_size,
            )

    def matrix_rotate(self, mat):
        self.X_axis = self.X_axis.matrix_rotate(mat)
        self.Y_axis = self.Y_axis.matrix_rotate(mat)
        self.Z_axis = self.Z_axis.matrix_rotate(mat)

    def translate(self, x=0, y=0, z=0):
        self.X_axis = self.X_axis.translate(x, y, z)
        self.Y_axis = self.Y_axis.translate(x, y, z)
        self.Z_axis = self.Z_axis.translate(x, y, z)

    def matrix_translate(self, mat):
        self.X_axis = self.X_axis.matrix_translate(mat)
        self.Y_axis = self.Y_axis.matrix_translate(mat)
        self.Z_axis = self.Z_axis.matrix_translate(mat)

    def translate_on_axis(self, mat):
        self.X_axis = self.X_axis.translate_on_axis(mat)
        self.Y_axis = self.Y_axis.translate_on_axis(mat)
        self.Z_axis = self.Z_axis.translate_on_axis(mat)

    def __str__(self):
        return f"X-Axis = {self.X_axis}\nY-Axis = {self.Y_axis}\nZ-Axis = {self.Z_axis}"

    def __repr__(self):
        return self.__str__()
