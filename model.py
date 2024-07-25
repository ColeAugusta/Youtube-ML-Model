import pandas as pd

if __name__ == "__main__":
    
    # load trending data
    data = pd.read_csv("csv/US_youtube_trending_data.csv")
    print(data.info())

    # drop unneeded columns from data:
    # eg. video and channel ids
    data = data.drop(["video_id", "channel_id", "category_id"], axis=1)

    # split into target and feature datasets
    features = data.columns.difference(["view_count"]) 
    X = data[features]
    Y = data["view_count"]

    
