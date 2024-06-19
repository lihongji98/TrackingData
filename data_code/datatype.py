from dataclasses import dataclass
from typing import List


@dataclass
class PlayerCoordinate:
    x : float
    y : float


@dataclass 
class PlayerVelocity:
    v_x: float
    v_y: float


@dataclass
class PlayerFrameInfo:
    number: str
    coordinate: PlayerCoordinate
    velocity: PlayerVelocity = None


@dataclass
class BallCoordinate:
    x: float
    y: float


@dataclass
class BallVelocity:
    v_x: float
    v_y: float


@dataclass
class BallFrameInfo:
    coordinate: BallCoordinate
    velocity: BallVelocity = None


@dataclass
class BallPossessionInfo:
    from_player: str
    to_player: str
    side: str


@dataclass
class FrameInfo:
    frame: int
    ball_info: BallFrameInfo
    home_player_info: List[PlayerFrameInfo]
    away_player_info: List[PlayerFrameInfo]
    ball_possession: BallPossessionInfo


@dataclass
class FrameInfoVis:
    frame: int
    ball_info: BallFrameInfo
    home_player_info: List[PlayerFrameInfo]
    away_player_info: List[PlayerFrameInfo]


