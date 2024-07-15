from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d


from src.helper.types import Array, Point
from src.plane.coordinate_system import CoordinateSystem, draw_line
from src.plane.math_functions import _cosine_degrees, _sine_degrees

from typing import Optional, List

from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import numpy as np

import config as CFG


class PaperPlane:
    sides: Array["4,3,3", int]
    marble: Array["3", int]
    translation_matrix = None
    rotation_matrix = None

    marble_cs: CoordinateSystem

    def __init__(
        self,
        plane_width=3.0,
        plane_length=7.0,
        plane_space=0.5,
        plane_depth=1.4,
        marble_x=3.5,
        marble_y=0,
        marble_z=1.3,
        marble_cs_size=4,
    ):
        marble = np.array([marble_x, marble_y, marble_z])
        self.initial_marble_z = marble_z

        self.threat = None
        self.translation_matrix = None
        self.rotation_matrix = None

        sides = np.zeros((4, 3, 3))

        sides[0] = np.array(
            [
                [0.0, -plane_width, 0.0],
                [plane_length, 0.0, 0.0],
                [0.0, -plane_space, 0.0],
            ]
        )

        sides[1] = np.array(
            [[0.0, plane_width, 0.0], [plane_length, 0.0, 0.0], [0.0, plane_space, 0.0]]
        )

        sides[2] = np.array(
            [[0.0, -plane_space, 0.0], [plane_length, 0.0, 0.0], [0, 0.0, plane_depth]]
        )

        sides[3] = np.array(
            [[0.0, plane_space, 0.0], [plane_length, 0.0, 0.0], [0, 0.0, plane_depth]]
        )

        self.rotation_matrix_from_global_cs = [
            [-0.0, 1.0, 0.0],
            [1.0, 0.0, -0.0],
            [-0.0, -0.0, -1.0],
        ]

        self.translation_matrix = [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]

        # Translate plane to center
        for s in sides:
            for c in s:
                c[0] = c[0] - (plane_length / 2)
                c[2] = c[2] - (plane_depth / 2)

        marble[0] = marble[0] - (plane_length / 2)
        marble[2] = marble[2] + (plane_depth / 2)

        self.marble_cs = CoordinateSystem(
            points=np.array(
                [
                    [[0, 0, 0], [marble_cs_size, 0, 0]],
                    [[0, 0, 0], [0, marble_cs_size, 0]],
                    [[0, 0, 0], [0, 0, marble_cs_size]],
                ]
            )
        )

        # Rotate marble_cs to be concurrent with the literature
        # 180 pitch yaw -90
        self.marble_cs.matrix_rotate(self.rotation_matrix_from_global_cs)

        # translate marble cs to be inline with the actual marble
        self.marble_cs.translate(x=0, y=0, z=marble[2])

        self.sides = sides

        # TEST rotate to bring in concurrent with literature
        for i, s in enumerate(self.sides):
            self.sides[i] = np.array(
                [np.matmul(self.rotation_matrix_from_global_cs, x) for x in s]
            )

        self.marble = marble

    def matrix_rotate(self, mat):
        self.reverse_translate()

        # TODO:
        # Sadly a better solution isn't immediately obvious to me and I'm in a hurry.
        # So I have to think about a more elegant solution to the whole matrix rotation thing
        # Right now I reverse the current rotation back to the global coordinate system
        # by multiplying the points with the transposed matrix, which is equal to the inverse
        # in the case of rotation matricies.
        # Then applying the rotation
        # And rotating the points back in the plane coordinate system.
        # Reverse translation

        # Reverse rotation #1
        for i, s in enumerate(self.sides):
            self.sides[i] = np.array(
                [
                    np.matmul(np.array(self.rotation_matrix_from_global_cs).T, x)
                    for x in s
                ]
            )
        # Rotate Plane itself
        for i, s in enumerate(self.sides):
            self.sides[i] = np.array([np.matmul(mat, x) for x in s])

        self.rotation_matrix = mat
        # Bring back plane in new coordinate system
        for i, s in enumerate(self.sides):
            self.sides[i] = np.array(
                [np.matmul(self.rotation_matrix_from_global_cs, x) for x in s]
            )

        # Rotate Marble – Again: back to the global referential system -> actual rotation -> back to plane system
        self.marble = np.matmul(
            np.array(self.rotation_matrix_from_global_cs).T, self.marble
        )
        self.marble = np.matmul(mat, self.marble)
        self.marble = np.matmul(
            np.array(self.rotation_matrix_from_global_cs), self.marble
        )

        # Rotate Marble CS
        self.marble_cs.matrix_rotate(np.array(self.rotation_matrix_from_global_cs).T)
        self.marble_cs.matrix_rotate(mat)
        self.marble_cs.matrix_rotate(np.array(self.rotation_matrix_from_global_cs))

        # TODO:
        # handle threat location after rotation

        # TODO: translate back to Plane System
        self.translate_back()

    def reverse_translate(self):
        mat = np.copy(self.translation_matrix)
        mat[0][3] = mat[0][3] * -1
        mat[1][3] = mat[1][3] * -1
        mat[2][3] = mat[2][3] * -1

        # Translate Plane
        self.matrix_translate(mat)
        # Translate Marble
        self.translate_matrix_marble(mat)
        # Translate Marble CS
        self.marble_cs.matrix_translate(mat)

    def translate_back(self):
        self.matrix_translate(self.translation_matrix)
        self.translate_matrix_marble(self.translation_matrix)
        self.marble_cs.matrix_translate(self.translation_matrix)

    def translate(self, x, y, z):
        self.translation_matrix[0][3] = self.translation_matrix[0][3] + x
        self.translation_matrix[1][3] = self.translation_matrix[1][3] + y
        self.translation_matrix[2][3] = self.translation_matrix[2][3] + z

        self.matrix_translate(self.translation_matrix)

        self.translate_matrix_marble(self.translation_matrix)

        self.marble_cs.matrix_translate(self.translation_matrix)

    def translate_matrix_marble(self, mat):
        nina = self.marble
        nina = np.append(nina, [1], axis=0)

        self.marble = np.matmul(mat, nina)[:-1]

    def matrix_translate(self, mat):
        # Translate Plane itself
        for i, s in enumerate(self.sides):
            nina = np.append(s, np.ones((len(s), 1)), axis=1)
            self.sides[i] = np.array([np.matmul(mat, x)[:-1] for x in nina])

    def get_angle_to_threat(self):
        """ """
        if self.threat is None:
            return

        T = self.threat.copy()
        M = self.marble.copy()
        Z = self.marble.copy()
        # Set the z point to 0
        Z[2] = 0
        X = self.marble_cs.X_axis.points[1].copy()

        n_a = np.cross(X - M, Z - M)
        n_b = np.cross(T - M, Z - M)

        dihedral_angle = np.dot(n_a, n_b) / (
            np.sqrt(np.sum(np.square(n_a))) * np.sqrt(np.sum(np.square(n_b)))
        )
        print(dihedral_angle)

        dihedral_angle = np.arccos(dihedral_angle)
        return dihedral_angle * 180 / np.pi

    def add_threat(
        self,
        bearing,
        horizontal_distance=5,
        threat_elevation_deg=None,
        threat_height_above_ground=0,
    ):
        """
        You can either set a specific elevation and bearing of the threat or just a bearing and a height above ground.

        Known Bearing ( 𝜃 ) = Angle from North
        Given a known Bearing ( 𝜃 ) and a horizontal distance (HzDist) from a known point (X0,Y0), the coordinates (X,Y) may be calculated as follows:

        X = [ Sin( 𝜃 ) x HzDist ] + X0
        Y = [ Cos( 𝜃 ) x HzDist ] + Y0 # X0 and Y0 are 0 in this case

        This works for ALL bearings 0°<360°

        The bearing is calculated from the base of the ground coordinate system translated to the marble hight and then rotated back to the plane
        """
        assert abs(bearing) < 360
        if bearing < 0:
            bearing = 360 + bearing

        # calc bearing Coords
        x = _cosine_degrees(bearing) * horizontal_distance
        y = _sine_degrees(bearing) * horizontal_distance

        coords = [x, y, self.initial_marble_z]
        print(coords)

        # rotate bearing coords
        if self.rotation_matrix is not None:
            coords = np.matmul(self.rotation_matrix, coords)
        coords = np.matmul(self.rotation_matrix_from_global_cs, coords)

        # translate bearing coords.
        coords = np.append(coords, [1], axis=0)
        coords = np.matmul(self.translation_matrix, coords)[:-1]

        # Now we have to calculate the angle or set the threat to a given height.
        if threat_elevation_deg is not None:
            # Simple geometry:
            # threat_height = (horizontal_distance * sin(Beta)) / cos Beta
            # threat_z = marble_z - threat_height
            threat_height = (
                horizontal_distance * np.sin(threat_elevation_deg * np.pi / 180)
            ) / np.cos(threat_elevation_deg * np.pi / 180)
            coords[2] = max(0, self.marble[2] - threat_height)
        else:
            coords[2] = threat_height_above_ground

        self.threat = coords.copy()

    def draw(
        self,
        ax,
        draw_marble=True,
        face_color=[0, 0.5, 0.9, 0.5],
        marble_color=[0.98, 0.52, 0, 1],
        marble_size=9,
        marble_cs_lw=2,
        plane_lw=0.9,
        show_marble_cs_labels=True,
        set_to_marble_height=True,
        shadow_floor=None,  # wo soll der Shatten hinfallen? bzw tiefster Punkt im Plot. If None, wird kein Schatten gemalt
    ):
        # plot the plane
        for s in self.sides:
            collection = Poly3DCollection(
                [s], linewidths=plane_lw, zorder=90, edgecolors=[0, 0, 0, 0.9]
            )
            collection.set_facecolor(face_color)
            ax.add_collection3d(collection)

        # plot the marble
        if draw_marble:
            ax.plot3D(
                xs=self.marble[0],
                ys=self.marble[1],
                zs=self.marble[2],
                zorder=100,
                color=marble_color,
                marker="o",
                linewidth=2,
                markersize=marble_size,
            )

            self.marble_cs.plot_coordinates(
                ax=ax, lw=marble_cs_lw, show_labels=show_marble_cs_labels
            )

        if self.threat is not None:
            # Since we don't know the elevation of a threat, we display it on the marble height
            # TODO: display it x degree down
            zs = self.marble[2] if set_to_marble_height else self.threat[2]

            ax.plot3D(
                xs=self.threat[0],
                ys=self.threat[1],
                zs=self.threat[2],
                zorder=100,
                color="red",
                marker="o",
                linewidth=2,
                markersize=marble_size,
            )

            draw_line(ax, self.marble, self.threat, color="r", lw=plane_lw)

        if shadow_floor is not None:
            for s in self.sides:
                ss = s.copy()

                for t in ss:
                    t[2] = shadow_floor

                collection = Poly3DCollection(
                    [ss], linewidths=0, zorder=0, edgecolors=[0, 0, 0, 0]
                )
                collection.set_facecolor(CFG.shadow_color)
                ax.add_collection3d(collection)
