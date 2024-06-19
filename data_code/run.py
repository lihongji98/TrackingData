from utils import connect, disconnect
from model import TrackingDataReader
from typing import List



if __name__ == "__main__":
    client = connect()
    tdr = TrackingDataReader(client=client, database="trackingdata", collection="Game_2")
    # home_pass_frames: List[FrameInfo] = tdr.get_frames(match_id="2", side="home", event_type="PASS")
    # away_pass_frames: List[FrameInfo] = tdr.get_frames(match_id="2", side="away", event_type="PASS")    
    # tdr.event_visualization(file_name="1", match_id="2", side="home", event_type="PASS")
    tdr.event_visualization(file_name="1", match_id="2", start_frame=12212 - 25 * 10, end_frame=12212, show_pitch_control=True)
    
    # data_list = [tdr.genenrate_frame_graph_data(frame) for frame in home_pass_frames] + [tdr.genenrate_frame_graph_data(frame) for frame in away_pass_frames]
    # torch.save(data_list, "train/game_2.pt")
    # data = tdr.genenrate_frame_graph_data(home_pass_frames[90])
    # print(data)
    
    # tdr.show_frame_pitch_control(home_pass_frames[105], granularity=1)

    disconnect()
    
"""
TODO:
1. velocity computing (done)
2. pitch control (done)
3. graph establishing (done)
4. pass predicting
"""