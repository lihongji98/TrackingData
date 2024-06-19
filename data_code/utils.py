from pymongo import MongoClient
from typing import List
from datatype import PlayerFrameInfo, PlayerVelocity, PlayerCoordinate, BallVelocity, BallFrameInfo, FrameInfoVis
import math
import numpy as np
import matplotlib.pyplot as plt


pitch_length, pitch_width = 105, 68


def connect():
    return MongoClient('mongodb://localhost:27017/')


def disconnect():
    return MongoClient('mongodb://localhost:27017/').close()


def add_frame_velocity(curr_frame: List, last_frame: List) -> List[BallFrameInfo | PlayerFrameInfo]:
    curr_ball_pos: BallFrameInfo = curr_frame[0] 
    last_ball_pos: BallFrameInfo = last_frame[0]
    if (math.isnan(curr_ball_pos.coordinate.x) or math.isnan(curr_ball_pos.coordinate.y) or \
        math.isnan(last_ball_pos.coordinate.x) or math.isnan(last_ball_pos.coordinate.y)):
        ball_velocity = BallVelocity(v_x=0, v_y=0)
    else:
        ball_v_x = round((curr_ball_pos.coordinate.x - last_ball_pos.coordinate.x) * pitch_length / 0.04, 5)
        ball_v_y = round((curr_ball_pos.coordinate.y - last_ball_pos.coordinate.y) * pitch_width / 0.04, 5)
        ball_velocity = BallVelocity(v_x=ball_v_x, v_y=ball_v_y)
    curr_ball_pos.velocity = ball_velocity

    curr_player_pos: List[PlayerFrameInfo] = curr_frame[1:]
    last_player_pos: List[PlayerFrameInfo] = last_frame[1:]

    for index, (cur_info, last_info) in enumerate(zip(curr_player_pos, last_player_pos)):
        assert cur_info.number == last_info.number, "The current player is matched to the player in the last frame, the list order goes wrong..."
        v_x = round((cur_info.coordinate.x - last_info.coordinate.x) * pitch_length / 0.04, 5)
        v_y = round((cur_info.coordinate.y - last_info.coordinate.y) * pitch_width / 0.04, 5)  
        curr_player_pos[index].velocity = PlayerVelocity(v_x=v_x, v_y=v_y)

    frame_info: List[BallFrameInfo | PlayerFrameInfo] = [curr_ball_pos] + curr_player_pos
    
    return frame_info


def compute_coordinate_distance(left_player: PlayerCoordinate, right_player: PlayerCoordinate):
    return np.sqrt(((left_player.x - right_player.x) * pitch_length)**2 + ((left_player.y - right_player.y) * pitch_width)**2)


def sigmoid(x):
    return 1/(np.exp(-x) + 1)


def plot_pitch(field_dimen=(105.0, 68.0), field_color='white', linewidth=2, markersize=20):
    fig, ax = plt.subplots(figsize=(12, 8))  # create a figure 

    # decide what color we want the field to be. Default is green, but can also choose white
    if field_color == 'green':
        ax.set_facecolor('mediumseagreen')
        lc = 'whitesmoke'  # line color
        pc = 'w'  # 'spot' colors
    elif field_color == 'white':
        lc = 'k'
        pc = 'k'

    # ALL DIMENSIONS IN m
    meters_per_yard = 0.9144  # unit conversion from yards to meters
    half_pitch_length = field_dimen[0] / 2.0  # length of half pitch
    half_pitch_width = field_dimen[1] / 2.0  # width of half pitch
    signs = [-1, 1]

    # Soccer field dimensions typically defined in yards, so we need to convert to meters
    goal_line_width = 8 * meters_per_yard
    box_width = 20 * meters_per_yard
    box_length = 6 * meters_per_yard
    area_width = 44 * meters_per_yard
    area_length = 18 * meters_per_yard
    penalty_spot = 12 * meters_per_yard
    corner_radius = 1 * meters_per_yard
    D_length = 8 * meters_per_yard
    D_radius = 10 * meters_per_yard
    D_pos = 12 * meters_per_yard
    centre_circle_radius = 10 * meters_per_yard

    # Define the translation offset to move the center to the top-left corner
    offset_x = half_pitch_length
    offset_y = half_pitch_width

    # Plot half way line and center circle
    ax.plot([offset_x, offset_x], [0, 2 * half_pitch_width], lc, linewidth=linewidth)
    ax.scatter(offset_x, offset_y, marker='o', facecolor=lc, linewidth=0, s=markersize)
    y = np.linspace(0, 2 * centre_circle_radius, 50)
    x = np.sqrt(centre_circle_radius**2 - (y - centre_circle_radius)**2)
    ax.plot(offset_x + x, offset_y - centre_circle_radius + y, lc, linewidth=linewidth)
    ax.plot(offset_x - x, offset_y - centre_circle_radius + y, lc, linewidth=linewidth)

    for s in signs:  # plots each line separately
        # Plot pitch boundary
        ax.plot([0, 2 * half_pitch_length], [offset_y + s * half_pitch_width, offset_y + s * half_pitch_width], lc, linewidth=linewidth * 2)
        ax.plot([offset_x + s * half_pitch_length, offset_x + s * half_pitch_length], [0, 2 * half_pitch_width], lc, linewidth=linewidth * 2)

        # Goal posts & line
        ax.plot([offset_x + s * half_pitch_length, offset_x + s * half_pitch_length], \
                [offset_y - goal_line_width / 2.0, offset_y + goal_line_width / 2.0], pc + 's', markersize=6 * markersize / 20.0, linewidth=linewidth)

        # 6 yard box
        ax.plot([offset_x + s * half_pitch_length, offset_x + s * (half_pitch_length - box_length)], \
                [offset_y + box_width / 2.0, offset_y + box_width / 2.0], lc, linewidth=linewidth)
        ax.plot([offset_x + s * half_pitch_length, offset_x + s * (half_pitch_length - box_length)], \
                [offset_y - box_width / 2.0, offset_y - box_width / 2.0], lc, linewidth=linewidth)
        ax.plot([offset_x + s * (half_pitch_length - box_length), offset_x + s * (half_pitch_length - box_length)], \
                [offset_y - box_width / 2.0, offset_y + box_width / 2.0], lc, linewidth=linewidth)

        # Penalty area
        ax.plot([offset_x + s * half_pitch_length, offset_x + s * (half_pitch_length - area_length)],\
                 [offset_y + area_width / 2.0, offset_y + area_width / 2.0], lc, linewidth=linewidth)
        ax.plot([offset_x + s * half_pitch_length, offset_x + s * (half_pitch_length - area_length)], \
                [offset_y - area_width / 2.0, offset_y - area_width / 2.0], lc, linewidth=linewidth)
        ax.plot([offset_x + s * (half_pitch_length - area_length), offset_x + s * (half_pitch_length - area_length)], \
                [offset_y - area_width / 2.0, offset_y + area_width / 2.0], lc, linewidth=linewidth)

        # Penalty spot
        ax.scatter(offset_x + s * (half_pitch_length - penalty_spot), offset_y, marker='o', facecolor=lc, linewidth=0, s=markersize)

        # Corner flags
        y = np.linspace(0, corner_radius, 50)
        x = np.sqrt(corner_radius**2 - y**2)
        ax.plot(offset_x + s * half_pitch_length - s * x, offset_y - half_pitch_width + y, lc, linewidth=linewidth)
        ax.plot(offset_x + s * half_pitch_length - s * x, offset_y + half_pitch_width - y, lc, linewidth=linewidth)

        # Draw the D
        y = np.linspace(-D_length, D_length, 50)
        x = np.sqrt(D_radius**2 - y**2) + D_pos
        ax.plot(offset_x + s * half_pitch_length - s * x, offset_y + y, lc, linewidth=linewidth)

    # Remove axis labels and ticks
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])

    # Set axis limits
    xmax = field_dimen[0] 
    ymax = field_dimen[1]
    ax.set_xlim([0, xmax])
    ax.set_ylim([0, ymax])
    ax.set_axisbelow(True)

    ax.invert_yaxis()

    return fig, ax


def frame_plot(event_frame_info, plot_buffer, fig, ax):
    ball_info: BallFrameInfo = event_frame_info.ball_info
    home_player_info: List[PlayerFrameInfo] = event_frame_info.home_player_info
    away_player_info: List[PlayerFrameInfo] = event_frame_info.away_player_info

    for scatter in plot_buffer:
        scatter.remove()
    plot_buffer = []

    palyer_size, ball_size = 120, 70
    # Plot the ball
    ball_x, ball_y = ball_info.coordinate.x, ball_info.coordinate.y
    ball_plot_buffer = ax.scatter(ball_x * 105, ball_y * 68, color='black', s=ball_size)
    plot_buffer.append(ball_plot_buffer)
    # Plot home players
    for player in home_player_info:
        player_x, player_y = player.coordinate.x, player.coordinate.y
        player_vx, player_vy = player.velocity.v_x, player.velocity.v_y

        home_player_plot_buffer = ax.scatter(player_x * 105, player_y * 68, s=palyer_size, color="red", alpha=0.7)
        plot_buffer.append(home_player_plot_buffer)
        home_player_velocity_buffer = ax.quiver(player_x * 105, player_y * 68, player_vx, -player_vy, color="red", 
                                                scale_units='inches', scale=10.,width=0.0015,headlength=5,headwidth=3,alpha=0.7)
        plot_buffer.append(home_player_velocity_buffer)

    # Plot away players
    for player in away_player_info:
        player_x, player_y = player.coordinate.x, player.coordinate.y
        player_vx, player_vy = player.velocity.v_x, player.velocity.v_y

        away_player_plot_buffer = ax.scatter(player_x * 105, player_y * 68, s=palyer_size, color="blue",  alpha=0.7)
        plot_buffer.append(away_player_plot_buffer)
        away_player_velocity_buffer = ax.quiver(player_x * 105, player_y * 68, player_vx, -player_vy, color="blue", 
                                                scale_units='inches', scale=10.,width=0.0015,headlength=5,headwidth=3,alpha=0.7)
        plot_buffer.append(away_player_velocity_buffer)
    
    return plot_buffer
