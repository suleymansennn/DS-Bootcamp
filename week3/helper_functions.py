import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Qt5Agg")

colors = ['#FFB6B9', '#FAE3D9', '#BBDED6', '#61C0BF', "#CCA8E9", "#F67280"]
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)

df = sns.load_dataset("titanic")


def check_df(dataframe, head=5, tail=5):
    print("*" * 70)
    print(" Shape ".center(70, "*"))
    print("*" * 70)
    print(dataframe.shape)

    print("*" * 70)
    print(" Types ".center(70, "*"))
    print("*" * 70)
    print(dataframe.dtypes)

    print("*" * 70)
    print(" Head ".center(70, "*"))
    print("*" * 70)
    print(dataframe.head(head))

    print("*" * 70)
    print(" Tail ".center(70, "*"))
    print("*" * 70)
    print(dataframe.tail(tail))

    print("*" * 70)
    print(" NA ".center(70, "*"))
    print("*" * 70)
    print(dataframe.isnull().sum())

    print("*" * 70)
    print(" Quantiles ".center(70, "*"))
    print("*" * 70)
    print(dataframe.describe([.01, .05, .1, .5, .9, .95, .99]).T)

    print("*" * 70)
    print(" Duplicate Rows ".center(70, "*"))
    print("*" * 70)
    print(dataframe.duplicated().sum())

    print("*" * 70)
    print(" Uniques ".center(70, "*"))
    print("*" * 70)
    print(dataframe.nunique())


check_df(df)


def grab_col_names(dataframe, cat_th=10, car_th=25):
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtype == "O"]
    cat_but_car = [col for col in dataframe.columns if
                   dataframe[col].nunique() > car_th and dataframe[col].dtype == "O"]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtype != "O"]
    num_but_cat = [col for col in dataframe.columns if
                   dataframe[col].dtype != "O" and dataframe[col].nunique() < cat_th]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    cat_cols = num_but_cat + cat_cols
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    print("".center(150, '~'))
    print(f"Observations: {dataframe.shape[0]}")
    print(f"Features: {dataframe.shape[1]}")
    print(f"Shape of dataset: {dataframe.shape}")
    print("".center(150, '~'))
    print(f"Categorical Variables: \n\n{cat_cols}")
    print("".center(150, '~'))
    print(f"\n\nNumerical Variables: \n\n{num_cols}")
    print("".center(150, '~'))
    print(f"\n\nNumber of Categorical Variables: \n\n{len(cat_cols)}")
    print("".center(150, '~'))
    print(f"\n\nNumber of Numerical Variables: \n\n{len(num_cols)}")
    print("".center(150, '~'))
    return cat_cols, num_cols, cat_but_car


def cat_summary(dataframe, cat_col, target, plot=True):

    print(pd.DataFrame({cat_col: dataframe[cat_col].value_counts(),
                        "Ratio": 100 * dataframe[cat_col].value_counts() / len(dataframe),
                        "Target_Median": dataframe.groupby(cat_col)[target].median(),
                        "Target_Mean": dataframe.groupby(cat_col)[target].mean(),
                       "Target_Sum:": dataframe.groupby(cat_col)[target].sum()}))
    print("".center(100, "#"))
    if target in cat_cols:
        sns.set_style("whitegrid")
        sns.countplot(data=dataframe, x=cat_col, hue=target, palette=colors)
        plt.title(f"Count plot of {cat_col.capitalize()}")
        plt.show(block=True)
    else:
        sns.set_style("whitegrid")
        sns.catplot(data=dataframe, x=cat_col, y=target, kind="box", palette=colors)
        plt.title(f"Box Plot of {cat_col.capitalize()}")
        plt.show(block=True)


cat_cols, num_cols, cat_but_car = grab_col_names(df)

for col in cat_cols:
    cat_summary(df, col, "survived")


def cat_plots(dataframe, cat_col):
    print("".center(100, "#"))
    print(dataframe[cat_col].value_counts())

    plt.figure(figsize=(15, 10))
    plt.suptitle(cat_col.capitalize(), size=16)
    plt.subplot(1, 2, 1)
    plt.title("Percentages")
    plt.pie(dataframe[cat_col].value_counts().values.tolist(),
            labels=dataframe[cat_col].value_counts().keys().tolist(),
            labeldistance=1.1,
            wedgeprops={'linewidth': 3, 'edgecolor': 'white'},
            colors=colors,
            autopct='%1.0f%%')

    plt.subplot(1, 2, 2)
    plt.title("Countplot")
    sns.countplot(data=dataframe, x=cat_col, palette=colors)
    plt.tight_layout(pad=3)
    plt.show(block=True)


for col in cat_cols:
    cat_plots(df, col)

def num_summary(dataframe, col_name, target):
    quantiles = [.01, .05, .1, .5, .9, .95, .99]
    print(dataframe[col_name].describe(quantiles).T)
    plt.figure(figsize=(15, 10))
    plt.suptitle(col_name.capitalize(), size=16)
    plt.subplot(1, 3, 1)
    plt.title("Histogram")
    sns.histplot(dataframe[col_name], color="#FFB6B9")
    plt.subplot(1, 3, 2)
    plt.title("Box Plot")
    sns.boxplot(data=dataframe, y=col_name, color="#F67280")
    if target in num_cols:
        plt.subplot(1, 3, 3)
        plt.title("Scatter Plot")
        sns.scatterplot(x=target, y=col_name, data=dataframe, color="#CCA8E9")
    else:
        plt.subplot(1, 3, 3)
        sns.histplot(data=dataframe, x=col_name,label=col_name, hue=target, kde=True, color=colors)
    plt.show(block=True)


for col in num_cols:
    num_summary(df, col, "survived")


def high_correlated_cols(dataframe, target, plot=True, corr_th=0.7):
    corr = dataframe.corr()
    corr_matrix = corr.abs()
    msk = np.triu(corr)
    drop_list = [col for col in num_cols if dataframe[[col, target]].corr().abs().loc[col, target] > corr_th]
    if plot:
        sns.heatmap(corr, mask=msk, annot=True, linecolor="black", cmap=colors)
        plt.yticks(rotation=0, size=10)
        plt.xticks(rotation=75, size=10)
        plt.title('\nCorrelation Map\n', size=15)
        plt.show()
    return drop_list

drop_list = high_correlated_cols(df, "survived")

