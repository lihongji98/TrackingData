from typing import List
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from datatype import PlayerCoordinate
import math


def get_tracking_document_to_update(csv_path, match_id, team_name, side):
    time = datetime.now()
    document_update = []

    with open(csv_path) as data:
        data = data.readlines()
        for index, line in enumerate(data):

            frame_document:dict[str, str | 
                                    dict[str, int | str] |
                                    List[dict[str, int | str | PlayerCoordinate]]] = {}

            metadata_dict = {}
            player_pos_list = []

            frame_document["timestamp"] = time + timedelta(seconds=index)

            if index < 2:
                pass

            elif index == 2: # basic info 
                raw_keys = line.strip().split(",")
                keys = [key for key in raw_keys if len(key) > 0]
                player_list = keys[3:]

            else:   # tracking data
                line = line.strip()
                frame_info = line.split(",")[0:3]
                player_frame_tracking = line.split(",")[3:-2]
                ball_frame_tracking = line.split(",")[-2:]

                period: int = int(frame_info[0])
                frame: int = int(frame_info[1])
                
                frame_document["frame"] = frame

                metadata_dict["match_id"] = match_id
                metadata_dict["team_name"] = team_name
                metadata_dict["side"] = side
                metadata_dict["period"] = period

                frame_document["metadata"] = metadata_dict
                
                ball_x, ball_y = float(ball_frame_tracking[0]), float(ball_frame_tracking[1])
                ball_x = min(ball_x, 1) if not math.isnan(ball_x) else ball_x
                ball_y = min(ball_y, 1) if not math.isnan(ball_y) else ball_y

                frame_document["ball_position"] = asdict(PlayerCoordinate(ball_x, ball_y))

                assert len(player_frame_tracking) % 2 == 0, "tracking data size is wrong..."
                for i in range(0,len(player_frame_tracking), 2):
                    one_player_info = {}

                    player_index = int(i/2)
                    if player_index < len(player_list) - 1:
                        player_number = player_list[player_index][6:]
                        one_player_info["number"] = player_number
                    
                    player_x, player_y = float(player_frame_tracking[i]), float(player_frame_tracking[i+1])
                    player_x = min(player_x, 1) if not math.isnan(player_x) else player_x 
                    player_y = min(player_y, 1) if not math.isnan(player_y) else player_y
                    player_coordinate = asdict(PlayerCoordinate(player_x, player_y))
                    one_player_info["coordinate"] = player_coordinate

                    player_pos_list.append(one_player_info)
                    
                frame_document["player_position"] = player_pos_list
                
                document_update.append(frame_document)
    
    return document_update


def get_event_document_to_update(csv_path="Sample_Game_1\Sample_Game_1_RawEventsData.csv", match_id=1):
    document_update = []
    with open(csv_path) as events:
        events = events.readlines()
        for index, event in enumerate(events):
            
            event_document = {}
            event_document["match_id"] = match_id

            event = event.strip().split(",")

            if index > 0:
                event = ["NaN" if len(e) == 0 else e for e in event]
                side = event[0]
                event_type, event_subtype = event[1], event[2]
                period = int(event[3])
                start_frame, end_frame = int(event[4]), int(event[6])
                from_player, to_player = event[8], event[9]

                start_x, start_y = float(event[10]), float(event[11])
                start_x = min(start_x, 1) if not math.isnan(start_x) else start_x 
                start_y = min(start_y, 1) if not math.isnan(start_y) else start_y 
                start_coordinate = asdict(PlayerCoordinate(start_x, start_y))

                end_x, end_y = float(event[12]), float(event[13])
                end_x = min(end_x, 1) if not math.isnan(end_x) else end_x 
                end_y = min(end_y, 1) if not math.isnan(end_y) else end_y 
                end_coordinate = asdict(PlayerCoordinate(end_x, end_y))

                event_document["side"] = side.lower()
                event_document["event_type"] = event_type
                event_document["event_subtype"] = event_subtype
                event_document["period"] = period
                event_document["start_frame"] = start_frame
                event_document["end_frame"] = end_frame
                event_document["from_player"] = from_player
                event_document["to_player"] = to_player
                event_document["start_coordinate"] = start_coordinate
                event_document["end_coordinate"] = end_coordinate

                document_update.append(event_document)
    
    return document_update

