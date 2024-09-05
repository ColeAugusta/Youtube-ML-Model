import pandas as pd
import matplotlib as mp
import statsmodels.formula.api as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


if __name__ == "__main__":
    
    # load trending data
    data = pd.read_csv("csv/US_youtube_trending_data.csv")

    # drop unneeded columns from data:
    # video and channel ids, thumbnail links, etc
    data = data.drop(["video_id", "title", "publishedAt", "channelId", "channelTitle", "categoryId",
                       "trending_date", "thumbnail_link"], axis=1)

    # convert needed columns into ints if required
    
    # convert tags to # of tags
    tag_counts = []
    for value in data["tags"]:
        tag_counts.append(value.count('|') + 1)
    
    data = data.drop("tags", axis=1)
    data["tags"] = tag_counts

    # convert comments disabled to 0/1
    comments_disabled = []
    for value in data["comments_disabled"]:
        if value == "False":
            comments_disabled.append(1)
        else:
            comments_disabled.append(0)

    data = data.drop("comments_disabled", axis=1)
    data["comments_disabled"] = comments_disabled

    # convert ratings disabled to 0/1
    ratings_disabled = []
    for value in data["ratings_disabled"]:
        if value == "False":
            ratings_disabled.append(1)
        else:
            ratings_disabled.append(0)

    data = data.drop("ratings_disabled", axis=1)
    data["ratings_disabled"] = ratings_disabled

    # convert description to length of desc
    descriptions = []
    for value in data["description"]:
        if isinstance(value, str):
            descriptions.append(len(value))
        else:
            descriptions.append(0)

    data = data.drop("description", axis=1)
    data["description"] = descriptions

    print(data.dtypes)


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
    print('predictions: ', predict)
    print('intercept: ', model.intercept_)
    print('coefficients: ', model.coef_)
    print('score: ', model.score(X, Y))


