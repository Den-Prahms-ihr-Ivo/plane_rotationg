import pytest

from src.plane.coordinate_system import CoordinateSystem
from src.plane.math_functions import yaw_pitch_roll_to_matrix

import numpy as np
import numpy.testing as npt


def test_coordinate_system():
    points = np.array(
        [[[0, 0, 0], [1, 0, 0]], [[0, 0, 0], [0, 1, 0]], [[0, 0, 0], [0, 0, 1]]]
    )
    cs = CoordinateSystem(points=points)

    # Tests, ob alles sauber läuft:
    # X-Achse
    npt.assert_array_equal(cs.X_axis.points[0], [0, 0, 0])
    npt.assert_array_equal(cs.X_axis.points[1], [1, 0, 0])
    # Y-Achse
    npt.assert_array_equal(cs.Y_axis.points[0], [0, 0, 0])
    npt.assert_array_equal(cs.Y_axis.points[1], [0, 1, 0])
    # Z-Achse
    npt.assert_array_equal(cs.Z_axis.points[0], [0, 0, 0])
    npt.assert_array_equal(cs.Z_axis.points[1], [0, 0, 1])


def test_rotation():
    """
    According to my textbook the following should be equal to
    [[ 0.    , -0.7071,  0.7071],
    [ 0.    ,  0.7071,  0.7071],
    [-1.    ,  0.    ,  0.    ]]
    """
    nina = yaw_pitch_roll_to_matrix(yaw=45, pitch=90, round_to_decimal_places=4)
    npt.assert_array_equal(
        nina, [[0.0, -0.7071, 0.7071], [0.0, 0.7071, 0.7071], [-1.0, 0.0, 0.0]]
    )


def test_rotate_ground_cs_1():
    # ####################
    # TestCase #1 90° Yaw
    # ####################
    cs = CoordinateSystem(
        points=np.array(
            [[[0, 0, 0], [1, 0, 0]], [[0, 0, 0], [0, 1, 0]], [[0, 0, 0], [0, 0, 1]]]
        )
    )
    rotation_matrix = yaw_pitch_roll_to_matrix(yaw=90, round_to_decimal_places=4)

    # Test Rotation Matrix
    npt.assert_array_equal(rotation_matrix, [[0, -1, 0], [1, 0, 0], [0, 0, 1]])

    # Rotate Coordinate System
    cs.matrix_rotate(rotation_matrix)

    # Y-Achse
    npt.assert_array_equal(cs.X_axis.points[0], [0, 0, 0])
    npt.assert_array_equal(cs.X_axis.points[1], [0, 1, 0])
    # Y-Achse
    npt.assert_array_equal(cs.Y_axis.points[0], [0, 0, 0])
    npt.assert_array_equal(cs.Y_axis.points[1], [-1, 0, 0])
    # Z-Achse
    npt.assert_array_equal(cs.Z_axis.points[0], [0, 0, 0])
    npt.assert_array_equal(cs.Z_axis.points[1], [0, 0, 1])


def test_rotate_ground_cs_2():
    # ####################
    # TestCase #2 90° Pitch
    # ####################
    cs = CoordinateSystem(
        points=np.array(
            [[[0, 0, 0], [1, 0, 0]], [[0, 0, 0], [0, 1, 0]], [[0, 0, 0], [0, 0, 1]]]
        )
    )
    rotation_matrix = yaw_pitch_roll_to_matrix(pitch=90, round_to_decimal_places=4)

    # Test Rotation Matrix
    npt.assert_array_equal(rotation_matrix, [[0, 0, 1], [0, 1, 0], [-1, 0, 0]])

    # Rotate Coordinate System
    cs.matrix_rotate(rotation_matrix)

    # Y-Achse
    npt.assert_array_equal(cs.X_axis.points[0], [0, 0, 0])
    npt.assert_array_equal(cs.X_axis.points[1], [0, 0, -1])
    # Y-Achse
    npt.assert_array_equal(cs.Y_axis.points[0], [0, 0, 0])
    npt.assert_array_equal(cs.Y_axis.points[1], [0, 1, 0])
    # Z-Achse
    npt.assert_array_equal(cs.Z_axis.points[0], [0, 0, 0])
    npt.assert_array_equal(cs.Z_axis.points[1], [1, 0, 0])


def test_rotate_ground_cs_3():
    # ####################
    # TestCase #3 90° Roll
    # ####################
    cs = CoordinateSystem(
        points=np.array(
            [[[0, 0, 0], [1, 0, 0]], [[0, 0, 0], [0, 1, 0]], [[0, 0, 0], [0, 0, 1]]]
        )
    )
    rotation_matrix = yaw_pitch_roll_to_matrix(roll=90, round_to_decimal_places=4)

    # Test Rotation Matrix
    npt.assert_array_equal(rotation_matrix, [[1, 0, 0], [0, 0, -1], [0, 1, 0]])

    # Rotate Coordinate System
    cs.matrix_rotate(rotation_matrix)

    # Y-Achse
    npt.assert_array_equal(cs.X_axis.points[0], [0, 0, 0])
    npt.assert_array_equal(cs.X_axis.points[1], [1, 0, 0])
    # Y-Achse
    npt.assert_array_equal(cs.Y_axis.points[0], [0, 0, 0])
    npt.assert_array_equal(cs.Y_axis.points[1], [0, 0, 1])
    # Z-Achse
    npt.assert_array_equal(cs.Z_axis.points[0], [0, 0, 0])
    npt.assert_array_equal(cs.Z_axis.points[1], [0, -1, 0])
