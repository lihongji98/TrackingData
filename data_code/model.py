from pymongo import mongo_client
from pymongo.cursor import Cursor
from pymongo.collection import Collection 
from typing import List
from datatype import (  FrameInfo, FrameInfoVis, BallPossessionInfo,
                        PlayerCoordinate, BallFrameInfo,
                        PlayerFrameInfo,  BallCoordinate,
                        PlayerVelocity,   BallVelocity)
from utils import *
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.animation as animation 

from tqdm import tqdm

import torch
from torch_geometric.data import Data


pitch_length, pitch_width = 105, 68


class TrackingDataReader:
    def __init__(self, client: mongo_client, database: str, collection: str):
        self.event_collection: Collection = client[database][collection + "_Event"]
        self.frame_collection: Collection = client[database][collection]

        self.match_id = None
        self.side = None
        self.event_type = None

    def get_frames(self, match_id: str, side: str, event_type: str = "PASS") -> List[FrameInfo]:
        self.match_id = match_id
        self.side = side
        self.event_type = event_type

        query = {"match_id": match_id, "side": side, "event_type": event_type}
        pass_events: Cursor = self.event_collection.find(query)

        pass_frames_info: List[FrameInfo] = []
        for index, pass_event in enumerate(pass_events):
            assert pass_event["start_coordinate"]['x'] != float('nan') and pass_event["start_coordinate"]['y'] != float('nan') and \
                pass_event["end_coordinate"]['x'] != float('nan') and pass_event["end_coordinate"]['y'] != float('nan')

            start_frame, end_frame = pass_event["start_frame"], pass_event["end_frame"]
            from_player, to_player = pass_event["from_player"][6:], pass_event["to_player"][6:]
            start_coordinate, end_coordinate = PlayerCoordinate(**pass_event["start_coordinate"]), PlayerCoordinate(**pass_event["end_coordinate"])

            start_frame_info: FrameInfo = self._get_frame_info(start_frame, from_player, to_player, side)
            # end_frame_info: FrameInfo = self._get_frame_info(end_frame, from_player, to_player, side)

            pass_frames_info.append(start_frame_info)

        pass_frames_info = sorted(pass_frames_info, key=lambda x: x.frame)

        return pass_frames_info

    def _get_frame_side_info(self, frame: int, match_id: str, side: str, query: dict =None) -> List[BallFrameInfo | PlayerFrameInfo]:
        if query is None:
            query = {"metadata.match_id": match_id, "metadata.side": side, "frame": frame} 
        frame = self.frame_collection.find_one(query)

        ball_pos: dict[int, float] = frame["ball_position"]
        player_pos: List[dict[str, str | float]] = frame["player_position"]

        frame_pos: List[BallFrameInfo | PlayerFrameInfo] = [BallFrameInfo(coordinate=BallCoordinate(**ball_pos))]
        # first 1 elemenet is Ball Pos, and the next 11 are Player Pos 

        num_player_on_pitch = 0
        for pos in player_pos:
            if not (math.isnan(pos["coordinate"]['x']) or math.isnan(pos["coordinate"]['y'])):
                num_player_on_pitch += 1
                pos = PlayerFrameInfo(number=pos["number"], coordinate=PlayerCoordinate(**pos["coordinate"]))
                frame_pos.append(pos)
        assert num_player_on_pitch == 11, "The number of player on the pitch exceeds 11..."
        assert len(frame_pos) == 12, "The number of player on the pitch is less than 11..."

        return frame_pos

    def _get_frame_info(self, frame_num: int, match_id: str, from_player: str=None, to_player: str=None, side: str=None, visualize=False):
        frame_info_home: List[BallFrameInfo | PlayerFrameInfo] = self._get_frame_side_info(frame_num, match_id, "home")
        frame_info_away: List[BallFrameInfo | PlayerFrameInfo] = self._get_frame_side_info(frame_num, match_id, "away")

        ball_info: BallFrameInfo = frame_info_home[0]
        
        last_frame = frame_num - 1
        if last_frame > 1:
            last_frame_info_home: List[BallFrameInfo | PlayerFrameInfo] = self._get_frame_side_info(last_frame, match_id, "home")
            last_frame_info_away: List[BallFrameInfo | PlayerFrameInfo] = self._get_frame_side_info(last_frame, match_id, "away")

            frame_info_home_vel = add_frame_velocity(frame_info_home, last_frame_info_home)
            frame_info_away_vel = add_frame_velocity(frame_info_away, last_frame_info_away)
            assert len(frame_info_home_vel) == (1 + 11) and len(frame_info_away_vel) == (1 + 11), \
                "The total number of frame info is not aligned with reality, which shoudl be 1 + 11..."
        else:
            for index in range(len(frame_info_home)):
                frame_info_home[index].velocity = BallVelocity(v_x=0, v_y=0) if index == 0 else PlayerVelocity(v_x=0, v_y=0)
            for index in range(len(frame_info_home)):
                frame_info_away[index].velocity = BallVelocity(v_x=0, v_y=0) if index == 0 else PlayerVelocity(v_x=0, v_y=0)
        
        home_player_info: List[PlayerFrameInfo] = frame_info_home[1:]
        away_player_info: List[PlayerFrameInfo] = frame_info_away[1:]
        if not visualize:
            frame_info: FrameInfo = FrameInfo(frame_num, ball_info, home_player_info, away_player_info, BallPossessionInfo(from_player, to_player, side)) 
        else:
            frame_info: FrameInfoVis = FrameInfoVis(frame_num, ball_info, home_player_info, away_player_info)

        return frame_info
    
    def get_pitch_control(self, frame_info: FrameInfo | FrameInfoVis, granularity: float=1) -> np.ndarray: 
        pitch_control = np.zeros(shape=(int(pitch_length * 1 / granularity), int(pitch_width * 1 / granularity)))

        frame_index: int = frame_info.frame
        ball_info: BallFrameInfo = frame_info.ball_info 
        home_player_info: List[PlayerFrameInfo] = frame_info.home_player_info
        away_player_info: List[PlayerFrameInfo] = frame_info.away_player_info

        for player_info in home_player_info:
            player_area_influence = self._get_area_influence(player_info, ball_info.coordinate, granularity)
            pitch_control += player_area_influence
        
        for player_info in away_player_info:
            player_area_influence = self._get_area_influence(player_info, ball_info.coordinate, granularity)
            pitch_control -= player_area_influence
        
        return sigmoid(pitch_control)

    def _get_area_influence(self, player_info: PlayerFrameInfo, ball_pos: BallCoordinate, granularity: float=1):
        player_pos = np.array([player_info.coordinate.x * pitch_length, player_info.coordinate.y * pitch_width])
        player_vel = np.array([player_info.velocity.v_x, player_info.velocity.v_y])
        player_vel_magnitude = np.sqrt(np.sum(player_vel.__pow__(2)))

        influence_center = player_pos + 0.5 * player_vel_magnitude
        velocity_theta = np.arctan2(player_info.velocity.v_x, player_info.velocity.v_y)
        distance_to_ball = np.sqrt(np.sum((player_pos - np.array([ball_pos.x * pitch_length, ball_pos.y * pitch_width])).__pow__(2)))
        influence_radius = 4 * np.exp(np.log(10/4) * (distance_to_ball/18.5)**2) if distance_to_ball <= 18.5 else 10

        velocity_ratio = (player_vel_magnitude / 13) ** 2

        length_array = np.linspace(0, pitch_length, int(105 * 1 / granularity))
        width_array = np.linspace(0, pitch_width, int(68 * 1 / granularity))
        length_grid, width_grid = np.meshgrid(length_array, width_array, indexing='ij')
        pitch_surface = np.stack((length_grid, width_grid), axis=-1)

        X = (pitch_surface - influence_center).reshape(int(105 * 1 / granularity), int(68 * 1/granularity), 2, 1)
        X_T = X.transpose(0, 1, 3, 2)

        R = np.array([[np.cos(velocity_theta), -np.sin(velocity_theta)],
                      [np.sin(velocity_theta),  np.cos(velocity_theta)]]).reshape(2, 2)
        R_inv = np.linalg.inv(R)

        Sx = (influence_radius + influence_radius * velocity_ratio) * 0.5
        Sy = (influence_radius - influence_radius * velocity_ratio) * 0.5

        S = np.array([[Sx, 0], 
                      [0, Sy]]).reshape(2, 2)
                      
        Cov = R @ S @ S @ R_inv
        Cov_inv = np.linalg.inv(Cov)

        center = np.array([0, 0]).reshape(2, 1)
        center_T = center.transpose()

        energy = np.einsum('ijkl, ll, ijlk -> ijk', X_T, Cov_inv, X) * (-0.5)
        center_energy = np.einsum('ij, jj, ji -> i', center_T, Cov_inv, center) * (-0.5)

        player_influence = np.exp(energy - center_energy)

        return np.squeeze(player_influence)

    def show_frame_pitch_control(self, frame: FrameInfo, granularity: float=0.1):
        PC = self.get_pitch_control(frame, granularity)
        PC = np.transpose(PC)


        vmin = np.min(PC)
        vmax = np.max(PC)
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        

        plt.figure(figsize=(10, 8)) 
        im = plt.imshow(PC, cmap='coolwarm', origin='upper', norm=norm, alpha=1)

        cbar = plt.colorbar(im)
        plt.xlim(0, int(pitch_length * 1 / granularity))
        plt.ylim(0, int(pitch_width * 1 / granularity))
        plt.gca().invert_yaxis()
        plt.show()

    def event_visualization(self, file_name, match_id: str, start_frame: int=None, end_frame: int=None, 
                            side: str=None, event_type: str = "PASS", time: float=20791*0.04,
                            show_pitch_control=False):
        self.match_id = match_id
        self.side = side
        self.event_type = event_type
        frame = int(time/0.04)

        if start_frame is None and end_frame is None:
            query = {"match_id": match_id, "side": side, "event_type": event_type}
            pass_events: Cursor = self.event_collection.find(query)

            start_frame: int = None
            end_frame: int = None
            for pass_event in pass_events:
                start_frame, end_frame = pass_event["start_frame"], pass_event["end_frame"]
                if start_frame == frame:
                    break

            assert start_frame is not None and end_frame is not None, "event is not found..."

        for frame_idx in range(start_frame, end_frame + 1):
            event_frame_info: FrameInfoVis = self._get_frame_info(frame_idx, match_id, visualize=True)

        ffmpeg_writer = animation.writers["ffmpeg"]
        metadata = dict(title="TrackingData", author="lihong")
        writer = ffmpeg_writer(fps=25, metadata=metadata)

        fig, ax = plot_pitch()
        fig.set_tight_layout(True)
        file_name = f"{file_name}" + ".mp4"
        with writer.saving(fig, file_name, 100):
            plot_buffer = []
            for frame_idx in tqdm(range(start_frame, end_frame + 1)):
                event_frame_info: FrameInfoVis = self._get_frame_info(frame_idx, match_id, visualize=True)
                plot_buffer = frame_plot(event_frame_info, plot_buffer, fig, ax)
                
                if show_pitch_control:
                    PC = self.get_pitch_control(event_frame_info)
                    PC = np.transpose(PC)
                    granularity = 1

                    vmin = np.min(PC)
                    vmax = np.max(PC)
                    norm = colors.Normalize(vmin=vmin, vmax=vmax)
                    
                    im = ax.imshow(PC, cmap='coolwarm', origin='upper', norm=norm, alpha=0.7)

                    plt.xlim(0, int(pitch_length * 1 / granularity))
                    plt.ylim(0, int(pitch_width * 1 / granularity))
                    plot_buffer.append(im)

                    ax.invert_yaxis()
                
                writer.grab_frame()

        plt.clf()
        plt.close(fig)  

    def genenrate_frame_graph_data(self, frame_info: FrameInfo) -> Data:
        ball_info: BallFrameInfo = frame_info.ball_info
        ball_possessor_number: str = frame_info.ball_possession.from_player
        ball_receiver_number: str = frame_info.ball_possession.to_player
        ball_side: str = frame_info.ball_possession.side
        
        home_player_info: List[PlayerFrameInfo] = frame_info.home_player_info
        away_player_info: List[PlayerFrameInfo] = frame_info.away_player_info

        ball_coordinate: BallCoordinate =ball_info.coordinate       
                
        x = torch.empty(size=(22, 6), dtype=torch.float)
        y = torch.zeros(size=(22,), dtype=torch.long)

        node_index = 0
        for player_info in home_player_info:
            ball_possession = 1 if player_info.number == ball_possessor_number and ball_side == "home" else 0
            x_temp = torch.empty(size=(6,), dtype=torch.long)
            x_temp[0] = player_info.coordinate.x
            x_temp[1] = player_info.coordinate.y
            x_temp[2] = player_info.velocity.v_x
            x_temp[3] = player_info.velocity.v_y
            x_temp[4] = ball_possession
            x_temp[5] = torch.sqrt(torch.tensor((ball_coordinate.x - player_info.coordinate.x)**2 + (ball_coordinate.y - player_info.coordinate.y)**2))
            if player_info.number == ball_receiver_number:
                y[node_index] = 1
            x[node_index] = x_temp
            node_index += 1

        for player_info in away_player_info:
            ball_possession = 1 if player_info.number == ball_possessor_number and ball_side == "away" else 0
            x_temp = torch.empty(size=(6,), dtype=torch.float)
            x_temp[0] = player_info.coordinate.x
            x_temp[1] = player_info.coordinate.y
            x_temp[2] = player_info.velocity.v_x
            x_temp[3] = player_info.velocity.v_y
            x_temp[4] = ball_possession
            x_temp[5] = torch.sqrt(torch.tensor((ball_coordinate.x - player_info.coordinate.x)**2 + (ball_coordinate.y - player_info.coordinate.y)**2))
            if player_info.number == ball_receiver_number:
                y[node_index] = 1.0
            x[node_index] = x_temp
            node_index += 1

        assert torch.sum(x[:, 4]) == 1, "more than one player is holding the ball..."
        assert x.size(0) == 22, "the node num is not equal to 11 + 11 = 22..."
        assert torch.sum(y) == 1, "more than one player receives the ball..."

        # fully connected undirected graph
        edge_index = torch.empty(size=(2, 231), dtype=torch.long)
        edge_attr = torch.empty(size=(231, 3), dtype=torch.float)
        edge_index_counter = 0
        for i in range(22):
            for j in range(i + 1, 22):
                edge_attr_temp = torch.empty(size=(3,), dtype=torch.long)
                player_left_pos: PlayerCoordinate = PlayerCoordinate(x[i][0], x[i][1])
                player_right_pos: PlayerCoordinate = PlayerCoordinate(x[j][0], x[j][1])
                distance = compute_coordinate_distance(player_left_pos, player_right_pos)
                edge_attr_temp[0] = distance
                if (i > 10 and j > 10) or (i <= 10 and j <= 10): # teammate
                    edge_attr_temp[1] = 0
                    edge_attr_temp[2] = 1
                else: # opponent
                    edge_attr_temp[1] = 1
                    edge_attr_temp[2] = 0

                edge_attr[edge_index_counter] = edge_attr_temp
                edge_index[0][edge_index_counter] = i
                edge_index[1][edge_index_counter] = j

                edge_index_counter += 1

        data: Data = Data(x=x, edge_index=edge_index.t().contiguous().T, edge_attr=edge_attr, y=y)

        return data