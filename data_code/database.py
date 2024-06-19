from pymongo import MongoClient
from load_data import get_tracking_document_to_update, get_event_document_to_update

    

if __name__ == "__main__":
    client = MongoClient('mongodb://localhost:27017/')
    db = client['trackingdata']

    collection = db.Game_2_Event

    # documents = get_tracking_document_to_update(csv_path="Sample_Game_2\Sample_Game_2_RawTrackingData_Home_Team.csv", 
    #                                    match_id="2",
    #                                    team_name="Real Madrid", 
    #                                    side="home")
    # collection.insert_many(documents, ordered=False)
    
    # documents = get_tracking_document_to_update(csv_path="Sample_Game_2\Sample_Game_2_RawTrackingData_Away_Team.csv",
    #                                    match_id="2",
    #                                    team_name="Barcelona", 
    #                                    side="away")
    # collection.insert_many(documents, ordered=False)

    documents = get_event_document_to_update(csv_path="Sample_Game_2\Sample_Game_2_RawEventsData.csv", match_id="2")
    collection.insert_many(documents)

    client.close()
