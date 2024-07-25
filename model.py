import pandas as pd
import matplotlib as mp
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


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

    # 80% training, 20% testing split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.80, random_state=1)

    # create model
    model = LinearRegression()
    model.fit(X_train, Y_train)

    # use to predict view_count variable
    predict = model.predict(X_test)


