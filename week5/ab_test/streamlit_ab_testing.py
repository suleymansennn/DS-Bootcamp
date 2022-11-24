import pandas as pd
import streamlit as st
from scipy.stats import shapiro, levene, ttest_ind, mannwhitneyu
import plotly.express as px
import statsmodels.api as sm

pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", lambda x: "%.5f" % x)

st.title("Automated AB Testing")
st.image("https://ceblog.s3.amazonaws.com/wp-content/uploads/2018/06/29173427/ab-testing-2.jpg")

st.header("""A/B testing is a tool that helps you to make decisions based on data. 
This is the tool that allows 
companies with digital products but actually not only 
digital products to understand whether the improvement that
thre’re trying to implement in their product is actually an 
improvement.
     """)
col1, col2 = st.columns(2)
with col1:
    st.header("A/B Test Answers")
    st.text("1. Which metrics or user behavior change and how much do they change?")
    st.text("2. Is the change reliable?")
with col2:
    st.header("Does not Answers")
    st.text("1. Why did the change occur?")
    st.text("2. Do the users like the change?")
st.header("Steps of the A/B Testing Process")
st.text("""1. Define a hypothesis
-	Assumption on how the change will affect the users""")
st.text("2. Determine how to measure succes of the feature")
st.text("""3.   a. Assumption of Normality 
    b. Assumption of Variance Homogenity 
""")
st.text("4. Perform a Statistical Test")

default = st.sidebar.checkbox("Do you want to use example file?", value=False,
                              help="Use built-in example file to demo the app")
if default:
    df = pd.read_csv("df.csv")

else:
    data = st.sidebar.file_uploader('Upload a CSV')
    df = data.getvalue()
    df = pd.read_csv(data)

ab = st.sidebar.selectbox("Pick AB Column", df.columns.tolist())
target = st.sidebar.selectbox("Pick Target", df.columns.tolist())
outlier = st.sidebar.radio("Do you want to replace outlier with thresholds?", ("Yes", "No"), horizontal=True)


def outlier_threshold(dataframe, variable, q1, q3):
    quartile1 = dataframe[variable].quantile(q1 / 100)
    quartile3 = dataframe[variable].quantile(q3 / 100)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable, q1, q3):
    low_limit, up_limit = outlier_threshold(dataframe, variable, q1, q3)
    dataframe[variable] = dataframe[variable].apply(
        lambda x: up_limit if x > up_limit else (low_limit if x < low_limit else x))


if outlier == "Yes":
    quantile1 = st.sidebar.slider("Low Limit:", 0, 30)
    quantile3 = 100 - quantile1
    replace_with_thresholds(df, target, quantile1, quantile3)

st.header("Data Preview")
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Data", "Shape", "NA", "Quantiles", "Types"])
with tab1:
    st.dataframe(df)
with tab2:
    st.write(df.shape)
with tab3:
    st.dataframe(df.isnull().sum())
with tab4:
    st.dataframe(df.describe([.01, .05, .1, .5, .9, .95, .99]).T)
with tab5:
    st.write(df.dtypes.astype(str))


def assumption_tests():
    st.header("1. Normality Assumption:")
    st.text("Ho: There is no statistically significant difference between the means of the two groups "
            "distribution")
    st.text("Ha: There is statistically significant difference between the means of the two groups")

    tab1, tab2 = st.tabs(["Histogram", "QQ Plot"])
    with tab1:
        fig = px.histogram(df, x=target, color=ab,
                           title=f"Histogram of {target.capitalize()} by {ab.capitalize()}",
                           marginal="box")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        fig = sm.qqplot(df[target], line="s")
        st.plotly_chart(fig)

    test_stat1, pvalue1 = shapiro(df.loc[df[ab] == df[ab].value_counts().keys()[0], target].dropna())
    test_stat2, pvalue2 = shapiro(df.loc[df[ab] == df[ab].value_counts().keys()[1], target].dropna())

    test_stat3, pvalue3 = levene(df.loc[df[ab] == df[ab].value_counts().keys()[0], target].dropna(),
                                 df.loc[df[ab] == df[ab].value_counts().keys()[1], target].dropna())
    if pvalue3 > 0.05:
        equal_var = True
    else:
        equal_var = False

    summarize_data(pvalue1, pvalue2)
    hypothesis_test(pvalue1, pvalue2, equal_var, pvalue3)


def summarize_data(group_a_pvalue, group_b_pvalue):
    if group_a_pvalue and group_b_pvalue > 0.05:
        st.text(f"The mean of {target} by groups: {df.groupby(ab)[target].mean()}\nStandard deviation: "
                f"{df.groupby(ab)[target].std()}")
    else:
        st.text(f"The median of {target} by groups: {df.groupby(ab)[target].median()}\n"
                f"Interquantile Range:\n{df.groupby(ab)[target].describe()}")


def hypothesis_test(group_a_pvalue, group_b_pvalue, equal_var, levene_result):
    st.subheader("1.1 Shapiro Wilk Test Results")
    st.success(f"Group {df[ab].value_counts().keys()[0]} p-value: {'%.4f' % group_a_pvalue}")
    st.success(f"Group {df[ab].value_counts().keys()[1]} p-value: {'%.4f' % group_b_pvalue}")

    if group_a_pvalue and group_b_pvalue > 0.05:
        test_stat, pvalue = ttest_ind(df.loc[df[ab] == df[ab].value_counts().keys()[0], target],
                                      df.loc[df[ab] == df[ab].value_counts().keys()[1], target], equal_var=equal_var)

        st.success(f"""Determining the distribution of the variable {target} was important for choosing an appropriate 
        statistical method. So a Shapiro-Wilk test was performed and did not show evidence of non-normality. Based on 
        this outcome(p-value > 0.05), and after visual examination of the histogram of {target} and the QQ plot, algorithm decided
        to use a parametric (ttest_ind) test. Also, the mean with with the standard deviation were used to summarize the 
        variable {target})
        """)
        st.header("2. Variance Assumption")
        st.text("Ho: The compared groups have equal variance.")
        st.text("Ha: The compared groups do not have equal variance.")
        st.success(f"Levene test p value => {'%.4f' % levene_result}")
        if equal_var == False:
            st.success(
                "The p-value of the Levene’s test is significant, suggesting that there is a significant difference "
                "between the variances of the two groups. Therefore, we’ll use the Weltch t-test, which doesn’t "
                "assume the equality of the two variances.")
        else:
            st.success(
                "The p-value of the Levene’s test is not significant, suggesting that there is not a significant "
                "difference between the variances of the two groups. Therefore, we’ll use a standard independent 2 "
                "sample test that assumes equal population variances")

        last_report(pvalue, "3", test_stat)
    else:
        test_stat, pvalue = mannwhitneyu(df.loc[df[ab] == df[ab].value_counts().keys()[0], target],
                                         df.loc[df[ab] == df[ab].value_counts().keys()[1], target])
        st.success(f"""Determining the distribution of the variable {target} was important for choosing an appropriate 
        statistical method. So a Shapiro-Wilk test was performed and showed that the distribution of {target} departed
        significantly from normality. Based on this outcome (p-value < 0.05), a non-parametric (mannwhitneyu) test was 
        used, and the median with interquantile range were used to summarize the variable {target}.
        """)
        last_report(pvalue, "2", test_stat)
    return pvalue


def plot_result(test_stat, result):
    col1, col2 = st.columns(2)
    with col1:
        st.text(f"""The Calculated T-Statistic:
{'%.4f' % test_stat}""")
    with col2:
        st.text(f"""Is Significant?
{result}""")
    fig = px.bar(df.groupby(ab).mean().reset_index(), y=target, color=ab,
                 title=f"Histogram of {target.capitalize()} by {ab.capitalize()}", )
    st.plotly_chart(fig)


def last_report(test_result, header, test_stat, ):
    st.header(f"{header}.   A/B Test Result")
    if test_result < 0.05:
        result = True
        plot_result(test_stat, result)
        st.success(f""":dart: p-value => {'%.4f' % test_result} Since the p-value is less than 0.05
        , the Ho hypothesis can be rejected. So, there is  statistically significant difference between, the
        {df[ab].value_counts().keys()[0]} group and 
        {df[ab].value_counts().keys()[1]} group""")
    else:
        result = False
        plot_result(test_stat, result)
        st.warning(f""":negative_squared_cross_mark: p-value => {'%.4f' % test_result} Since the p-value is not less than 0.05
        , the Ho hypothesis cannot be rejected. So, there is no statistically significant difference between, the
        {df[ab].value_counts().keys()[0]} group and 
        {df[ab].value_counts().keys()[1]} group""")


button = st.sidebar.button(label="Calculate")
if button:
    assumption_tests()
