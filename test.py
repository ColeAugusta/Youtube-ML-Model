import pandas as pd

if __name__ == "__main__":
    
    data = pd.read_csv("US_youtube_trending_data.csv")
    print(data.info())
    print(data.head())