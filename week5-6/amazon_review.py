import pandas as pd
import math
import scipy.stats as st
import datetime as dt

pd.set_option("display.max_columns", None)
pd.set_option("display.expand_frame_repr", None)
pd.set_option("display.float_format", lambda x: "%.5f" % x)

df = pd.read_csv("measurement_problems/amazon_review.csv")
df.head()
df.shape
df.loc[df["reviewText"].isnull()]

df.groupby("asin").agg({"overall": "mean"})

df["reviewTime"] = pd.to_datetime(df["reviewTime"])
current_date = df["reviewTime"].max()
df["days"] = (current_date - df["reviewTime"]).dt.days
df.tail()

df["days"].describe([.1, .25, .5, .7, .8, .9, .95, .99]).T
df["days_cat"] = pd.qcut(df["days"], q=4, labels=["q1", "q2", "q3", "q4"])
df.groupby("days_cat").agg({"overall": "mean"}).mean()

df["helpful_no"] = df["total_vote"] - df["helpful_yes"]

True + True & False + (True + False & True + False) ** 5

def score_up_down_diff(up, down):
    return up - down

def score_average_rating(up, down):
    if up + down == 0:
        return 0
    return up / (up+down)

def wilson_lower_bound(up, down, confidence=0.95):
    """
    Wilson Lower Bound Score hesapla

    - Bernoulli parametresi p için hesaplanacak güven aralığının alt sınırı WLB skoru olarak kabul edilir.
    - Hesaplanacak skor ürün sıralaması için kullanılır.
    - Not:
    Eğer skorlar 1-5 arasıdaysa 1-3 negatif, 4-5 pozitif olarak işaretlenir ve bernoulli'ye uygun hale getirilebilir.
    Bu beraberinde bazı problemleri de getirir. Bu sebeple bayesian average rating yapmak gerekir.

    Parameters
    ----------
    up: int
        up count
    down: int
        down count
    confidence: float
        confidence

    Returns
    -------
    wilson score: float

    """
    n = up + down
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * up / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)

df = df[["overall", "reviewTime", "day_diff", "helpful_yes", "helpful_no","total_vote"]]
df.head()

df["score_pos_neg_diff"] = df.apply(lambda x: score_up_down_diff(x["helpful_yes"], x["helpful_no"]), axis=1)
df["score_average_rating"] = df.apply(lambda x: score_average_rating(x["helpful_yes"], x["helpful_no"]), axis=1)
df["wilson_lower_bound"] = df.apply(lambda x: wilson_lower_bound(x["helpful_yes"], x["helpful_no"]), axis=1)

df.head()

df.sort_values("wilson_lower_bound", ascending=False).head(20)

df.drop(["asin", "helpful", "summary", "unixReviewTime", "days_cat", "days"], axis=1, inplace=True)
df.head()

df.to_csv("amazon_sentiment.csv", index=False)