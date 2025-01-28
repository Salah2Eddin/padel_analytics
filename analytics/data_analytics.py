from typing import Optional
from dataclasses import dataclass
from copy import deepcopy
import pandas as pd
import numpy as np


class InvalidDataPoint(Exception):
    pass


# TODO: print to log file

@dataclass
class Position:
    """
        Position (meters) in a given frame
    """
    position: tuple[float, float]

    def __post_init__(self):
        assert isinstance(self.position[0], float)
        assert isinstance(self.position[1], float)


@dataclass
class PlayerPosition(Position):
    """
    Player position (meters) in a given frame
    """
    id: int

    @property
    def key(self) -> str:
        return f"player{self.id}"


@dataclass
class BallPosition(Position):
    """
    Ball position (meters) in a given frame
    """

    @property
    def key(self) -> str:
        return f"Ball"


@dataclass
class FrameData:
    """
    Tracker objects data collected in a given frame

    Attributes: 
        frame: frame of interest
        players_positions: players positions (meters) in the given frame
        ball_position: ball position (meters) in the given frame
    """

    frame: int = None
    players_positions: list[PlayerPosition] = None
    ball_position: BallPosition = None

    def validate(self) -> None:
        if self.frame is None:
            raise InvalidDataPoint("Unknown frame")

        # Players positions validation
        if self.players_positions is None:
            print("data_analytics: WARNING(Missing players position)")
            return None

        players_ids = []
        for i, player_pos in enumerate(deepcopy(self.players_positions)):
            player_id = player_pos.id

            if player_id in (1, 2, 3, 4):
                players_ids.append(player_id)
            else:
                del self.players_positions[i]

        if len(players_ids) != len(set(players_ids)):
            raise InvalidDataPoint("Duplicate player id")

        if len(self.players_positions) != 4:
            number_players_missing = 4 - len(self.players_positions)
            print(f"{number_players_missing} player/s missing")

        # Ball position validation
        if self.ball_position is None:
            print(f"ball is missing")

    def add_player_position(self, player_position: PlayerPosition):
        if self.players_positions is None:
            self.players_positions = [player_position]
        else:
            self.players_positions.append(player_position)

    # TODO: check if necessary
    def sort_players_positions(self) -> Optional[list[PlayerPosition]]:
        if self.players_positions:
            players_position = sorted(
                self.players_positions,
                key=lambda x: x.id,
            )
            return players_position

        print("data_analytics: impossible to sort, missing players position")
        return None

    def set_ball_position(self, ball_position: BallPosition):
        self.ball_position = ball_position


class DataAnalytics:
    """
    Tracker objects data collector 
    """

    def __init__(self):
        self.frames = [0]
        self.frames_data: list[FrameData] = []
        self.current_frame_data = FrameData(frame=self.frames[-1])

    def restart(self) -> None:
        self.__init__()

    @classmethod
    def from_dict(cls, data: dict):
        frames = data["frame"]
        instance = cls()
        instance.frames = frames

        frames_data = []
        for i in range(len(frames)):
            frame = frames[i]
            players_position = []
            for player_id in (1, 2, 3, 4):
                if (
                        data[f"player{player_id}_x"][i] is None
                        or
                        data[f"player{player_id}_y"][i] is None
                ):
                    continue

                players_position.append(
                    PlayerPosition(
                        id=player_id,
                        position=(
                            data[f"player{player_id}_x"][i],
                            data[f"player{player_id}_y"][i],
                        )
                    )
                )

            ball_position = BallPosition(
                position=(
                    data[f"ball_x"],
                    data[f"ball_y"]
                )
            )

            frames_data.append(
                FrameData(
                    frame=frame,
                    players_positions=players_position if players_position else None,
                    ball_position=ball_position,
                )
            )

        instance.frames_data = frames_data
        instance.current_frame_data = None

        return instance

    def into_dict(self) -> dict[str, list]:
        data = {
            "frame": [],
            "player1_x": [],
            "player1_y": [],
            "player2_x": [],
            "player2_y": [],
            "player3_x": [],
            "player3_y": [],
            "player4_x": [],
            "player4_y": [],
            "ball_x": [],
            "ball_y": [],
        }

        for frame_data in self.frames_data:
            data["frame"].append(frame_data.frame)
            number_frames = len(data["frame"])

            players_positions = frame_data.sort_players_positions()
            if players_positions:
                for player_position in players_positions:
                    data[f"{player_position.key}_x"].append(
                        player_position.position[0]
                    )
                    data[f"{player_position.key}_y"].append(
                        player_position.position[1]
                    )

            ball_position = frame_data.ball_position
            if ball_position:
                data["ball_x"].append(
                    ball_position.position[0]
                )
                data["ball_y"].append(
                    ball_position.position[1]
                )

            # Append missing values
            for k, v in data.items():
                if len(v) < number_frames:
                    data[k].append(None)
                    # data[k].append(data[k][-1] if len(data[k]) > 0 else None)

        print("data_analytics: missing values")
        for k, v in data.items():
            print(f"data_analytics: {k} - {len([v for x in v if x is None])}/{len(v)}")

        return data

    def __len__(self) -> int:
        return len(self.frames)

    def update(self):
        self.current_frame_data.validate()
        self.frames_data.append(self.current_frame_data)
        self.current_frame_data = FrameData(frame=self.frames[-1])

    def step(self, x: int = 1) -> None:
        new_frame = self.frames[-1] + 1

        assert new_frame not in self.frames

        self.frames.append(new_frame)
        self.update()

    def add_player_position(
            self,
            id: int,
            position: tuple[float, float],
    ):
        self.current_frame_data.add_player_position(
            PlayerPosition(
                id=id,
                position=position,
            )
        )

    def set_ball_position(
            self,
            position: tuple[float, float],
    ):
        self.current_frame_data.set_ball_position(
            BallPosition(
                position=position,
            )
        )

    def into_dataframe(self, fps: int) -> pd.DataFrame:
        """
        Retrieves a dataframe with additional features
        """

        player_ids = (1, 2, 3, 4)

        df = pd.DataFrame(self.into_dict())
        df["time"] = df["frame"] * (1 / fps)

        # Time in seconds between each frame for a given frame interval
        df[f"delta_time"] = df["time"].diff()

        df.ffill()

        for player_id in player_ids:
            for pos in ("x", "y"):
                # Displacement in x and y for each of the players
                df[f"player{player_id}_delta{pos}"] = df[f"player{player_id}_{pos}"].diff()

                # Velocity in x and y for each of the players
                # for a given time interval
                df[f"player{player_id}_V{pos}"] = df[f"player{player_id}_delta{pos}"] / df["delta_time"]

                # Velocity difference in x and y for each of the players
                # for a given time interval
                df[f"player{player_id}_deltaV{pos}"] = df[f"player{player_id}_V{pos}"].diff()

                # Acceleration in x and y for each of the players
                # for a given time interval
                df[f"player{player_id}_A{pos}"] = df[f"player{player_id}_deltaV{pos}"] / df["delta_time"]

            # Calculate player distance in between frames
            df[f"player{player_id}_distance"] = np.sqrt(
                df[f"player{player_id}_deltax"] ** 2 + df[f"player{player_id}_deltay"] ** 2
            )

            # Calculate accumulative sum of distances for each player
            df[f"player{player_id}_total_distance"] = df[f"player{player_id}_distance"].cumsum()

            # Calculate norm velocity for each player for a given time interval
            df[f"player{player_id}_Vnorm"] = np.sqrt(
                df[f"player{player_id}_Vx"] ** 2 + df[f"player{player_id}_Vy"] ** 2
            )

            # Calculate norm acceleration for each player for a given time interval
            df[f"player{player_id}_Anorm"] = np.sqrt(
                df[f"player{player_id}_Ax"] ** 2 + df[f"player{player_id}_Ay"] ** 2
            )

        for pos in ("x", "y"):
            # Displacement in x and y for each of the ball
            df[f"ball_delta{pos}"] = df[f"ball_{pos}"].diff()

            # Velocity in x and y for each of the ball
            # for a given time interval
            df[f"ball_V{pos}"] = df[f"ball_delta{pos}"] / df["delta_time"]

            # Velocity difference in x and y for each of the ball
            # for a given time interval
            df[f"ball_deltaV{pos}"] = df[f"ball_V{pos}"].diff()

            # Acceleration in x and y for each of the ball
            # for a given time interval
            df[f"ball_A{pos}"] = df[f"ball_deltaV{pos}"] / df["delta_time"]

        # Calculate ball distance in between frames
        df[f"ball_distance"] = np.sqrt(
            df[f"ball_deltax"] ** 2 + df[f"ball_deltay"] ** 2
        )

        # Calculate accumulative sum of distances for each player
        df[f"ball_total_distance"] = df[f"ball_distance"].cumsum()

        # Calculate norm velocity for the ball for a given time interval
        df[f"ball_Vnorm"] = np.sqrt(
            df[f"ball_Vx"] ** 2 + df[f"ball_Vy"] ** 2
        )

        # Calculate norm acceleration for the ball for a given time interval
        df[f"ball_Anorm"] = np.sqrt(
            df[f"ball_Ax"] ** 2 + df[f"ball_Ay"] ** 2
        )

        # Calculate ball direction
        df["ball_direction"] = 180 + np.arctan2(df["ball_deltay"], df["ball_deltax"]) / np.pi * 180

        # 1 if bounce
        dir_y = df.shift(1)["ball_direction"] // 90

        dir_x = df["ball_direction"] // 90
        dir_x_pos = (df["ball_direction"] + 10) // 90
        dir_x_neg = (df["ball_direction"] - 10) // 90

        df["ball_bounce"] = np.where(((abs(dir_y - dir_x_pos) == abs(dir_y - dir_x_neg)) & (abs(dir_y - dir_x) != 0)),
                                     "1", "")

        return df
