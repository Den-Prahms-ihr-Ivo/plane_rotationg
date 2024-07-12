import numpy as np

from src.helper.types import Array


def _cosine_degrees(degrees, decimal_places=4):
    return round(np.cos(degrees * np.pi / 180.0), decimal_places)


def _sine_degrees(degrees, decimal_places=4):
    return round(np.sin(degrees * np.pi / 180.0), decimal_places)


def yaw_pitch_roll_to_matrix(
    yaw: float = 0.0, pitch: float = 0.0, roll: float = 0.0, round_to_decimal_places=4
) -> Array[3, float]:

    # PHI
    sin_phi: float = _sine_degrees(roll, decimal_places=round_to_decimal_places)
    cos_phi: float = _cosine_degrees(roll, decimal_places=round_to_decimal_places)
    # THETA
    sin_theta: float = _sine_degrees(pitch, decimal_places=round_to_decimal_places)
    cos_theta: float = _cosine_degrees(pitch, decimal_places=round_to_decimal_places)
    # PSI
    sin_psi: float = _sine_degrees(yaw, decimal_places=round_to_decimal_places)
    cos_psi: float = _cosine_degrees(yaw, decimal_places=round_to_decimal_places)

    A = np.zeros((3, 3))

    A[0][0] = cos_theta * cos_psi
    A[0][1] = sin_phi * sin_theta * cos_psi - cos_phi * sin_psi
    A[0][2] = sin_phi * sin_psi + cos_phi * sin_theta * cos_psi

    A[1][0] = cos_theta * sin_psi
    A[1][1] = cos_phi * cos_psi + sin_phi * sin_theta * sin_psi
    A[1][2] = cos_phi * sin_theta * sin_psi - sin_phi * cos_psi

    A[2][0] = -sin_theta
    A[2][1] = sin_phi * cos_theta
    A[2][2] = cos_phi * cos_theta

    return A
