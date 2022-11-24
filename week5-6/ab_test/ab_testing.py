import pandas as pd
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from scipy.stats import shapiro, levene, ttest_ind


matplotlib.use("Qt5Agg")

pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", lambda x: "%.5f" % x)

colors = ['#FFB6B9', '#FAE3D9', '#BBDED6', '#61C0BF', "#CCA8E9", "#F67280"]

df_control = pd.read_excel("ab_test/ab_testing.xlsx", sheet_name="Control Group")
df_test = pd.read_excel("ab_test/ab_testing.xlsx", sheet_name="Test Group")


df_control.head()
df_test.head()

def num_summary(dataframe, col_name, target):
    quantiles = [.01, .05, .1, .5, .9, .95, .99]
    print("#" * 70)
    print(dataframe[col_name].describe(percentiles=quantiles))
    print("#" * 70)

    plt.figure(figsize=(15, 10))
    plt.suptitle(col_name.capitalize(), size=16)
    plt.subplot(1, 3, 1)
    plt.title("Histogram")
    sns.histplot(dataframe[col_name], color="#FFB6B9")

    plt.subplot(1, 3, 2)
    plt.title("Box Plot")
    sns.boxplot(data=dataframe, y=col_name, color="#F67280")

    plt.subplot(1, 3, 3)
    sns.scatterplot(data=dataframe, x=col_name, y=target, palette=colors, estimator=np.mean)
    plt.title(f"Average of {col_name.capitalize()} by {target.capitalize()}")
    plt.tight_layout(pad=1.5)
    plt.show(block=True)

for col in df_control.columns:
    num_summary(df_control, col, "Purchase")

for col in df_control.columns:
    num_summary(df_test, col, "Purchase")

df = pd.concat([df_control, df_test], ignore_index=True)
df.loc[0:39, "Control-Test"] = "Control"
df.loc[40:80, "Control-Test"] = "Test"
df.to_csv("df", index=False)
df["Control-Test"].value_counts()



# Gruplar arasında istatiski olarak fark yoktur.
# Ho: M1=M2
# Ha: M1!=M2
df.groupby("Control-Test").agg({"Purchase": "mean"})

test_stat, pvalue = shapiro(df.loc[df["Control-Test"]=="Control", "Purchase"])
print("Test Stat = %.4f, pvalue = %.4f" % (test_stat, pvalue))
# Ho red edilemez
# Normal dağılıma uygundur


test_stat, pvalue = shapiro(df.loc[df["Control-Test"]=="Test", "Purchase"])
print("Test Stat = %.4f, pvalue = %.4f" % (test_stat, pvalue))

# Ho red edilemez
# Normal dağılıma uygundur

test_stat, pvalue = levene(df.loc[df["Control-Test"]=="Control", "Purchase"],
                           df.loc[df["Control-Test"]=="Test", "Purchase"])
print("Test Stat = %.4f, pvalue = %.4f" % (test_stat, pvalue))
# Ho red edilemez
# Varyans homojenliği vardır

test_stat, pvalue = ttest_ind(df.loc[df["Control-Test"]=="Control", "Purchase"],
                                 df.loc[df["Control-Test"]=="Test", "Purchase"], equal_var=True)

print("Test Stat = %.4f, pvalue = %.4f" % (test_stat, pvalue))
# Ho red edilemez
# Gruplar arasında fark yoktur.

"""Eklenen ortalama teklif verme sistemini yeniden incelemek için daha fazla örnek içeren yeni bir test yapılabilir. Test ettiğimiz yeni sistem, ortalama kazançlarda önemli bir fark yaratmadı. Şimdilik uygulamaya koymak anlamsız."""