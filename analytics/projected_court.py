from copy import deepcopy
from typing import Literal, Optional
from dataclasses import dataclass
import cv2
import numpy as np
import supervision as sv

from constants import BASE_LINE, SIDE_LINE, SERVICE_SIDE_LINE, NET_SIDE_LINE
from utils import convert_meters_to_pixel_distance, convert_pixel_distance_to_meters
from trackers import Player, Players, Keypoint, Keypoints, Ball, Projection
from analytics.data_analytics import DataAnalytics

from . import SIZE_MULTIPLIER


class InconsistentPredictedKeypoints(Exception):
    pass


PointPixels = tuple[int, int]


@dataclass
class Rectangle:
    """
    Rectangle geometry utilities
    """

    top_left: PointPixels
    bottom_right: PointPixels

    @property
    def width(self) -> int:
        return self.bottom_right[0] - self.top_left[0]

    @property
    def height(self) -> int:
        return self.bottom_right[1] - self.top_left[1]

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def perimeter(self) -> int:
        return 2 * self.width + 2 * self.height


@dataclass
class ProjectedCourtKeypoints:
    """
    Projected court 12 points of interest 

        k11--------------------k12
        |                       |
        k8-----------k9--------k10
        |            |          |
        |            |          |
        |            |          |
        k6----------------------k7
        |            |          |
        |            |          |
        |            |          |
        k3-----------k4---------k5
        |                       |
        k1----------------------k2

    """

    k1: PointPixels
    k2: PointPixels
    k3: PointPixels
    k4: PointPixels
    k5: PointPixels
    k6: PointPixels
    k7: PointPixels
    k8: PointPixels
    k9: PointPixels
    k10: PointPixels
    k11: PointPixels
    k12: PointPixels

    def __post_init__(self):
        # Calculate reference origin
        self.origin = self._get_origin()

    @property
    def width(self) -> int:
        return self.k7[0] - self.k6[0]

    @property
    def height(self) -> int:
        return self.k1[1] - self.k11[1]

    def _get_origin(self) -> PointPixels:
        """
        Get the reference origin to estimate relative positions in meters
        """
        origin = (
            (self.k7[0] + self.k6[0]) // 2,
            (self.k7[1] + self.k6[1]) // 2,
        )

        return origin

    def keypoints(
            self,
            number_keypoints: Literal[12, 18, 22],
    ) -> list[Keypoint]:
        keypoints_12 = [
            Keypoint(id=i, xy=v)
            for i, (k, v) in enumerate(self.__dict__.items())
            if "k" in k
        ]

        extra_keypoints = []
        assert len(keypoints_12) == 12

        if number_keypoints == 18:
            extra_keypoints = [
                self.__getitem__("k1"),
                self.__getitem__("k2"),
                self.__getitem__("k6"),
                self.__getitem__("k7"),
                self.__getitem__("k11"),
                self.__getitem__("k12"),
            ]
        elif number_keypoints == 22:
            extra_keypoints = [
                self.__getitem__("k1"),
                self.__getitem__("k2"),
                self.__getitem__("k3"),
                self.__getitem__("k5"),
                self.__getitem__("k6"),
                self.__getitem__("k7"),
                self.__getitem__("k8"),
                self.__getitem__("k10"),
                self.__getitem__("k11"),
                self.__getitem__("k12"),
            ]

        keypoints_total = keypoints_12 + extra_keypoints

        return keypoints_total

    def __getitem__(self, k: str) -> Keypoint:
        idx = int(k.replace("k", "")) - 1
        return Keypoint(
            id=idx,
            xy=self.__dict__[k],
        )

    def lines(self) -> list[tuple[PointPixels, PointPixels]]:
        """
        Court lines start and end keypoints
        """
        return [
            (self.k1, self.k2),
            (self.k3, self.k5),
            (self.k6, self.k7),
            (self.k8, self.k10),
            (self.k11, self.k12),
            (self.k1, self.k11),
            (self.k4, self.k9),
            (self.k2, self.k12),
        ]

    def get_origin_shift(self):
        """
        Gets the shift in x and y directions to make a point relative to the origin of the court
        Returns:

        """
        return -self.origin[0], -self.origin[1]

    def shift_point_origin(
            self,
            point: tuple[float, float],
            dimension: Literal["pixels", "meters"],
    ) -> tuple[float, float]:
        """
        Change the origin of the point to the middle of the 
        projected court. If dimension is meters the vector entries
        are converted from pixels to meters
        """

        shifted_point = (
            float(point[0] - self.origin[0]),
            float(point[1] - self.origin[1]),
        )

        if dimension == "meters":
            shifted_point = tuple(
                convert_pixel_distance_to_meters(
                    pixel_distance=p,
                    reference_in_meters=BASE_LINE,
                    reference_in_pixels=self.width,
                )
                for p in shifted_point
            )

        return shifted_point


class ProjectedCourt:
    """
    Projected court abstraction with utilities to project and draw
    objects of interest in a 2d plane.

    Attributes:
        video_info: video information of interest
    """

    BUFFER = 50
    PADDING = 20
    ALPHA = 0.5

    def __init__(self, video_info: sv.VideoInfo):
        self.video_info = video_info
        # Canvas background points in pixels
        self.WIDTH = int(SIZE_MULTIPLIER[0] * video_info.width)
        self.HEIGHT = int(SIZE_MULTIPLIER[1] * video_info.height)

        self._set_canvas_background_position()
        self._set_projected_court_position()
        self._set_projected_court_keypoints()

        # Initialize the homography matrix H
        self.H = None

    def _set_canvas_background_position(self) -> None:

        """
        Set the canvas background position relative to the video frame
        """

        end_x = self.video_info.width - self.BUFFER
        end_y = self.BUFFER + self.HEIGHT
        start_x = end_x - self.WIDTH
        start_y = end_y - self.HEIGHT

        self.background_position = Rectangle(
            top_left=(int(start_x), int(start_y)),
            bottom_right=(int(end_x), int(end_y))
        )

    def _set_projected_court_position(self) -> None:

        """
        Set the projected court position (respecting measurements in meters) 
        inside the canvas background
        """

        court_start_x = self.background_position.top_left[0] + self.PADDING
        court_start_y = self.background_position.top_left[1] + self.PADDING
        court_end_x = self.background_position.bottom_right[0] - self.PADDING
        court_width = court_end_x - court_start_x
        court_height = convert_meters_to_pixel_distance(
            SIDE_LINE,
            reference_in_meters=BASE_LINE,
            reference_in_pixels=court_width
        )
        court_end_y = court_start_y + court_height

        self.court_position = Rectangle(
            top_left=(int(court_start_x), int(court_start_y)),
            bottom_right=(int(court_end_x), int(court_end_y)),
        )

    def _set_projected_court_keypoints(self) -> None:

        """
        Set the projected court 12 points of interest
        """

        service_line_height = convert_meters_to_pixel_distance(
            SERVICE_SIDE_LINE,
            reference_in_meters=BASE_LINE,
            reference_in_pixels=self.court_position.width,
        )

        self.court_keypoints = ProjectedCourtKeypoints(
            k1=(
                self.court_position.top_left[0],
                self.court_position.bottom_right[1],
            ),
            k2=self.court_position.bottom_right,
            k3=(
                self.court_position.top_left[0],
                self.court_position.bottom_right[1] - service_line_height,
            ),
            k4=(
                int(self.court_position.top_left[0] + self.court_position.width / 2),
                self.court_position.bottom_right[1] - service_line_height,
            ),
            k5=(
                self.court_position.bottom_right[0],
                self.court_position.bottom_right[1] - service_line_height,
            ),
            k6=(
                self.court_position.top_left[0],
                int(self.court_position.top_left[1] + self.court_position.height / 2),
            ),
            k7=(
                self.court_position.bottom_right[0],
                int(self.court_position.top_left[1] + self.court_position.height / 2),
            ),
            k8=(
                self.court_position.top_left[0],
                self.court_position.top_left[1] + service_line_height,
            ),
            k9=(
                int(self.court_position.top_left[0] + self.court_position.width / 2),
                self.court_position.top_left[1] + service_line_height,
            ),
            k10=(
                self.court_position.bottom_right[0],
                self.court_position.top_left[1] + service_line_height,
            ),
            k11=self.court_position.top_left,
            k12=(
                self.court_position.bottom_right[0],
                self.court_position.top_left[1],
            )
        )

    def draw_background_single_frame(self, frame: np.ndarray) -> np.ndarray:

        """
        Draw the projected court background on the given frame
        """

        shapes = np.zeros_like(frame, np.uint8)
        cv2.rectangle(
            shapes,
            self.background_position.top_left,
            self.background_position.bottom_right,
            (255, 255, 255),
            -1,
        )
        output_frame = frame.copy()
        mask = shapes.astype(bool)
        output_frame[mask] = cv2.addWeighted(
            output_frame,
            self.ALPHA,
            shapes,
            1 - self.ALPHA,
            0,
        )[mask]

        return output_frame

    def draw_projected_court_single_frame(self, frame: np.ndarray) -> np.ndarray:

        """
        Draw mini court points of interest and lines
        """

        for k, v in self.court_keypoints.__dict__.items():
            if "k" in k:
                cv2.circle(
                    frame,
                    v,
                    5,
                    (255, 0, 0),
                    -1,
                )
            else:
                cv2.circle(
                    frame,
                    v,
                    5,
                    (0, 255, 0),
                    -1,
                )

        for line in self.court_keypoints.lines():
            start_point = line[0]
            end_point = line[1]
            cv2.line(
                frame,
                start_point,
                end_point,
                (0, 0, 0),
                2,
            )

        return frame

    def homography_matrix(self, keypoints_detection: Keypoints) -> np.ndarray:

        """
        Calculates the homography matrix that projects the court keypoints detected
        on a given frame into the 2d court

        Parameters:
            keypoints_detection: predicted keypoints on a single frame
        """

        keypoints_detection = keypoints_detection.keypoints
        if len(keypoints_detection) == 12:
            src_keypoints = keypoints_detection
            # Court keypoints of the given frame
            src_points = np.array(
                [
                    keypoint.xy
                    for keypoint in src_keypoints
                ]
            )
            dst_keypoints = self.court_keypoints.keypoints(
                number_keypoints=12,
            )
            # Projected court keypoints
            dst_points = np.array(
                [
                    keypoint.xy
                    for keypoint in dst_keypoints
                ]
            )
        elif len(keypoints_detection) == 18:
            src_keypoints = keypoints_detection
            # Court keypoints of the given frame
            src_points = np.array(
                [
                    keypoint.xy
                    for keypoint in src_keypoints
                ]
            )
            dst_keypoints = self.court_keypoints.keypoints(
                number_keypoints=18,
            )
            # Projected court keypoints
            dst_points = np.array(
                [
                    keypoint.xy
                    for keypoint in dst_keypoints
                ]
            )
        elif len(keypoints_detection) == 22:
            src_keypoints = keypoints_detection
            # Court keypoints of the given frame
            src_points = np.array(
                [
                    keypoint.xy
                    for keypoint in src_keypoints
                ]
            )
            dst_keypoints = self.court_keypoints.keypoints(
                number_keypoints=22,
            )
            # Projected court keypoints
            dst_points = np.array(
                [
                    keypoint.xy
                    for keypoint in dst_keypoints
                ]
            )
        else:
            raise ValueError("Unhandled number of keypoints detected")

        if src_points.shape != dst_points.shape:
            raise InconsistentPredictedKeypoints("Don't have enough source points")

        # print("Source Keypoints: ")
        # print(src_keypoints)
        # print("Destination Keypoints: ")
        # print(dst_keypoints)
        # print("-"*20)

        H, _ = cv2.findHomography(src_points, dst_points)

        return H

    def point_pixel_distance_to_meters(self, point: tuple[int, int]) -> tuple[float, float]:
        """

        Args:
            point:

        Returns:

        """

        mx = convert_pixel_distance_to_meters(
            pixel_distance=point[0],
            reference_in_meters=BASE_LINE,
            reference_in_pixels=self.court_keypoints.width,
        )

        my = convert_pixel_distance_to_meters(
            pixel_distance=point[1],
            reference_in_meters=BASE_LINE,
            reference_in_pixels=self.court_keypoints.width,
        )

        return mx, my

    def project_player(
            self,
            player: Player,
    ) -> Player:
        """
        Player 2d court projection
        """

        projected_player = deepcopy(player)
        projected_player.projection = Projection.from_original(
            point=player.feet,
            homography_matrix=self.H,
        )

        return projected_player

    def project_ball(
            self,
            ball: Ball,
    ) -> Ball:
        """
        Ball 2d court projection
        """

        projected_point = Projection.from_original(
            point=ball.asint(),
            homography_matrix=self.H,
        )

        projected_ball = deepcopy(ball)
        projected_ball.projection = projected_point

        return projected_ball

    def draw_projected_player_and_collect_data(
            self,
            frame: np.ndarray,
            player: Player,
            data_analytics: DataAnalytics = None,
    ) -> np.ndarray:
        """
        Project and draw a single player
        """

        projected_player = self.project_player(player=player)

        if data_analytics is not None:
            shifted_player_projection = projected_player.projection.shift(
                self.court_keypoints.get_origin_shift()
            )

            shifted_player_projection_m = self.point_pixel_distance_to_meters(shifted_player_projection.tuple())

            # TODO: move this
            data_analytics.add_player_position(
                id=projected_player.id,
                position=shifted_player_projection_m,
            )

        return projected_player.draw_projection(frame)

    def draw_projected_players_and_collect_data(
            self,
            frame: np.ndarray,
            players: Players,
            data_analytics: DataAnalytics = None,
    ) -> np.ndarray:
        """
        Project and draw players
        """

        for player in players:
            frame = self.draw_projected_player_and_collect_data(
                frame=frame,
                player=player,
                data_analytics=data_analytics,
            )

        return frame

    def draw_projected_ball_and_collect_data(
            self,
            frame: np.ndarray,
            projected_ball: Ball,
            data_analytics: DataAnalytics = None,
    ) -> np.ndarray:
        """
        Project and draw ball
        """

        projected_ball = self.project_ball(ball=projected_ball)

        if data_analytics is not None:
            shifted_ball_projection = projected_ball.projection.shift(
                self.court_keypoints.get_origin_shift()
            )

            shifted_ball_projection_m = self.point_pixel_distance_to_meters(shifted_ball_projection.tuple())

            # TODO: move this
            data_analytics.set_ball_position(
                position=shifted_ball_projection_m,
            )

        return projected_ball.draw_projection(frame)

    def draw_projections_and_collect_data(
            self,
            frame: np.ndarray,
            keypoints_detection: Keypoints,
            players: Optional[Players],
            ball: Optional[Ball],
            data_analytics: Optional[DataAnalytics] = None,
            is_fixed_keypoints: bool = False,
    ) -> tuple[np.ndarray, DataAnalytics]:
        """
        Project and draw court and objects of interest.
        Collect objects of interest data.

        Parameters:
            frame: video frame 
            keypoints_detection: court keypoints detection
            players: players bounding box detection
            ball: ball position
            data_analytics: instance for data collection
            is_fixed_keypoints: True if the keypoints detection is fixed
        """

        output_frame = self.draw_background_single_frame(frame)
        output_frame = self.draw_projected_court_single_frame(output_frame)

        if self.H is None:
            if keypoints_detection:
                print("projected_court: First homography matrix calculation ...")
                self.H = self.homography_matrix(keypoints_detection)
                print("projected_court: Done.")
        else:
            if not is_fixed_keypoints:
                if keypoints_detection:
                    print("projected_court: Homography matrix calculation ...")
                    self.H = self.homography_matrix(keypoints_detection)
                    print("projected_court: Done.")
                else:
                    # Can't calculate homography for this frame
                    print("projected_court: Missing keypoints for homography calculation")
                    self.H = None

        if self.H is not None and players is not None:
            output_frame = self.draw_projected_players_and_collect_data(
                output_frame,
                players=players,
                data_analytics=data_analytics,
            )
        else:
            print("projected_court: Missing data for players projection")

        if self.H is not None and ball:
            output_frame = self.draw_projected_ball_and_collect_data(
                output_frame,
                projected_ball=ball,
                data_analytics=data_analytics
            )
        else:
            print("projected_court: Missing data for ball projection")

        return output_frame, data_analytics
