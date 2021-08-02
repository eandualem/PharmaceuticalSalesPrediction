import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

stores_types = ["Store type A", "Store type B", "Store type C", "Store type D"]
state_holidays = ["Normal Day", "Public holiday", "Easter holiday", "Christmas"]
school_holidays = ["School is closed", "School is open"]
assortment = ["Basic", "Extra", "Extended"]
promo = ["Not participating", "Participating"]
store_open = ["Closed", "Open"]


@st.cache
def loadData():
  df_store = pd.read_csv("./pages/store.csv")
  df_train = pd.read_csv("./pages/train.csv")
  df_train_store = pd.merge(df_train.reset_index(), df_store, how='inner', on='Store')
  df_train_store['Date'] = pd.DatetimeIndex(df_train_store['Date'])
  df_train_store['Year'] = pd.DatetimeIndex(df_train_store['Date']).year
  df_train_store['Month'] = pd.DatetimeIndex(df_train_store['Date']).month
  df_train_store['Day'] = pd.DatetimeIndex(df_train_store['Date']).day
  return df_train_store


def plot_trend(df, columns, feature, title, x_label="", y_label="", labels=['']):
  fig = plt.figure(figsize=(18, 6))
  for i in range(len(columns)):
    sns.lineplot(x=df.index, y=df[columns[i]][feature], label=labels[i])
  plt.title(title, fontsize=15, fontweight='bold')
  plt.ylabel(x_label, fontsize=14)
  plt.xlabel(y_label, fontsize=14)
  st.pyplot(fig)


def seasonality(df):
  st.header("Seasonality in the data")

  df["StoreType"] = df["StoreType"].apply(lambda x: stores_types[x])
  daily_trend = df.groupby(['Day', 'StoreType']).agg({'Customers': 'mean', 'Sales': 'mean'})
  daily_trend = daily_trend.unstack().swaplevel(0, 1, 1).sort_index(1)
  columns = ["Store type A", "Store type B", "Store type C", "Store type D"]
  plot_trend(daily_trend, columns, 'Sales', 'Average daily sales for 3 years', labels=columns)
  st.write("Here we can see there is a very similar trend in through a month between the stores. Store B has more sales but still, the shape of the graph is similar.")

  plot_trend(daily_trend, columns, 'Customers', 'Average daily customers for 3 years', labels=columns)
  st.write('We can observe there is a similarity between the average sales and the average number of customers.')

  monthly_trend = df.groupby(['Month', 'StoreType']).agg({'Customers': 'mean', 'Sales': 'mean'})
  monthly_trend = monthly_trend.unstack().swaplevel(0, 1, 1).sort_index(1)
  plot_trend(monthly_trend, columns, 'Sales', 'Average monthly sales for 3 years', labels=columns)
  plot_trend(monthly_trend, columns, 'Customers', 'Average monthly sales for 3 years', labels=columns)
  st.write("The highlights sales are in April, July, and December. These are the holidays in Germany, where the stores are located. Sales increase in all stores leading to this holidays and there is a decrease in sales after they pass.")


def correlation(df):
  st.header("Correlation between features")
  corr_all = df[['Open', 'Promo', 'Promo2', 'Sales', 'Customers', 'DayOfWeek', 'StateHoliday', 'SchoolHoliday', 'StoreType',
                 'Assortment', 'CompetitionDistance', 'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear']].corr()
  mask = np.zeros_like(corr_all, dtype=np.bool)
  mask[np.triu_indices_from(mask)] = True
  fig, ax = plt.subplots(figsize=(12, 12))
  heatmap = sns.heatmap(corr_all, mask=mask, square=True, linewidths=.5,
                        vmin=-1, vmax=1, cmap='coolwarm', annot=True, fmt='.2f')
  heatmap.set_title('Correlation between features', fontdict={'fontsize': 15}, pad=12)
  st.pyplot(fig)
  st.write("There is a strong positive correlation between sales and Customers. Promo has a positive correlation with sales and customers. State holidays and Days of the week have a negative correlation with sales and customers.")


def promotions(df):
  st.header("Impact of promotion on sales and customers")
  df["Promo"] = df["Promo"].apply(lambda x: promo[x])
  df["Promo2"] = df["Promo2"].apply(lambda x: promo[x])
  fp = sns.factorplot(data=df, x='Month', y="Sales", palette='plasma', hue='Year',
                      row='Promo', col='Promo2', color='green', size=4, aspect=2)
  fp.fig.subplots_adjust(top=.9)
  fp.fig.suptitle("The impact of promo and promo2 on seles each year", fontsize=15, fontweight='bold')
  st.pyplot(fp)

  fp = sns.factorplot(data=df, x='Month', y="Customers", palette='plasma', hue='Year',
                      row='Promo', col='Promo2', color='green', size=4, aspect=2)
  fp.fig.subplots_adjust(top=.9)
  fp.fig.suptitle("The impact of promo and promo2 on average number of customers in each year",
                  fontsize=15, fontweight='bold')
  st.pyplot(fp)
  st.write("The most interesting thing about the data was that running a promotion for the second time didn’t help in increasing sales. It is probably because customers already purchased whatever they wanted during the first promotional sale.")


def get_assortment(df):
  st.header("Impact of assortment on sales and customers")
  df["Assortment"] = df["Assortment"].apply(lambda x: assortment[x])
  fig, (axis1, axis2) = plt.subplots(1, 2, figsize=(15, 4))
  sns.barplot(x='Assortment', y='Sales', data=df, ax=axis1)
  sns.barplot(x='Assortment', y='Customers', data=df, ax=axis2)
  axis1.set_title("Assortment Vs Sales ")
  axis2.set_title("Assortment Vs Customers ")
  st.pyplot(fig)
  st.write("Assortment type extra has the highest sales, and customers love it. It has more than double the average customer count than basic and extended.")


def app():
  st.title('Insights')
  df = loadData().copy()

  st.sidebar.title('Feature To Explore')
  col = st.sidebar.selectbox("Choose feature of data", (["Seasonality", "Correlation", "Promotions", "Assortment"]))
  if col == "Seasonality":
    seasonality(df)
  elif col == "Correlation":
    df = loadData().copy()
    correlation(df)
  elif col == "Promotions":
    promotions(df)
  else:
    get_assortment(df)
