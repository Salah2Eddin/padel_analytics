import numpy as np
from dataclasses import dataclass


def project_point(point: tuple[float, float], homography_matrix: np.ndarray):
    """
    Project point given a homography matrix H.
    Args:
        point: point to be projected
        homography_matrix: homography matrix that projects into the 2d plane
    Returns:
        projected point
    """
    assert homography_matrix.shape == (3, 3)

    src_point = np.array([point[0], point[1], 1])

    dst_point = np.matmul(homography_matrix, src_point)
    dst_point = dst_point / dst_point[2]

    return dst_point[0], dst_point[1]

@dataclass
class Projection:
    """
    Holds the x, y data for the projection of a point on the 2d court
    Attributes:
        x: projected x position
        y: projected y position
    """
    x: float
    y: float

    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    @classmethod
    def from_original(cls, point: tuple[float, float], homography_matrix: np.ndarray):
        """
        Creates an instance from the original point using the homography matrix
        Args:
            point: original point
            homography_matrix: homography matrix that projects into the 2d plane
        """
        x, y = project_point(point, homography_matrix)

        return cls(x, y)

    @classmethod
    def from_tuple(cls, point: tuple[float, float]):
        """
        Creates an instance of Projection from a tuple
        Args:
            point: point tuple
        """
        return cls(point[0], point[1])

    def shift(self, xy_shift):
        """
        Returns a new Projection instance shifted by the given x shift and y shift
        Args:
            xy_shift: shift in x, y direction
        """
        return Projection(self.x + xy_shift[0], self.y + xy_shift[1])

    def tuple(self):
        """
        returns projection as a tuple
        """
        return self.x, self.y

    def asint(self):
        return int(self.x), int(self.y)

    def __repr__(self):
        return self.x, self.y