import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

plt.style.use('fivethirtyeight')
from matplotlib.pylab import rcParams

rcParams['figure.figsize'] = 28, 18
import statsmodels.api as sm
# from statsmodels.tsa.stattools import adfuller
# from statsmodels.tsa.seasonal import seasonal_decompose
import itertools
import warnings

warnings.filterwarnings("ignore")
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签

# df1 = pd.read_csv("sales_data.csv", encoding="gbk")
dateparse = lambda x: pd.to_datetime(x, format='%Y%m', errors='coerce')
df = pd.read_csv("sales_data.csv", parse_dates=['month'], index_col='month', date_parser=dateparse, encoding="gbk")
ts = df[pd.Series(pd.to_datetime(df.index, errors='coerce')).notnull().values]

# grouped = ts.groupby(["city", "product"])  # 按城市产品分组
# rcParams['figure.figsize'] = 28, 18
# fig, ax = plt.subplots()
# for city_product, group in grouped:
#     group.plot(y='quantity', label=city_product, ax=ax, title=u'2014-2016各城市各产品销量', fontsize=25)
#     ax.set_xlabel(u'时间（月）')
#     ax.set_ylabel(u'销量（个）')
#     ax.title.set_size(28)
#     ax.xaxis.label.set_size(25)
#     ax.yaxis.label.set_size(25)
#     ax.legend(loc='best', fontsize=18)
# plt.show()

# fig, axes = plt.subplots(2,3, figsize = (30, 20)) #绘制前2个城市，3种产品
# for (city_product, group), ax in zip(grouped, axes.flatten()):
#     group.plot(y = 'quantity',ax = ax, title = str(city_product), fontsize = 25)
#     ax.set_xlabel(u'时间（月）')
#     ax.set_ylabel(u'销量（个）')
#     ax.xaxis.label.set_size(23)
#     ax.yaxis.label.set_size(23)
# plt.show()


list1 = list(ts['city_product'].unique())
# df1["month"] = df1["month"].apply(lambda x: datetime.strptime(str(x), '%Y%m'))
# ds = df1
# for x in list1[0:3]:  # 展示第一座城市的三个产品销量的时序走势
#     ds = df1[df1['city_product'] == x]
#     plt.figure(figsize=(20, 10), dpi=100)
#     plt.rcParams['font.sans-serif'] = ['SimHei']
#     ds.plot('month', 'quantity', legend=True, title=x)
#     plt.show()

# 网格搜寻方法找到合适的SARIMA模型并预测
df_forecast = pd.DataFrame(columns=('城市&产品类型', '预测值', '下限', '上限'))
ts1 = ts.groupby(['city_product', ts.index])['quantity'].sum().unstack(level=0)
for x in list1:
    y = ts1[x]
    p = d = q = range(0, 2)
    pdq = list(itertools.product(p, d, q))
    pdq_x_PDQs = [(x[0], x[1], x[2], 0) for x in list(itertools.product(p, d, q))]
    a = []
    b = []
    c = []
    wf = pd.DataFrame()
    for param in pdq:
        for seasonal_param in pdq_x_PDQs:
            try:
                mod = sm.tsa.statespace.SARIMAX(y, order=param, seasonal_order=seasonal_param,
                                                enforce_stationarity=False, enforce_invertibility=False)
                results = mod.fit()
                # print('ARIMA{}x{} - AIC:{}'.format(param, seasonal_param, results.aic))
                a.append(param)
                b.append(seasonal_param)
                c.append(results.aic)
            except:
                continue
    wf['pdq'] = a
    wf['pdq_x_PDQs'] = b
    wf['aic'] = c

    row_index = wf[wf['aic'] == wf['aic'].min()].index.tolist()[0]
    print(row_index)
    mod = sm.tsa.statespace.SARIMAX(y,
                                    order=wf.iloc[row_index, 0],
                                    seasonal_order=wf.iloc[row_index, 1],
                                    enforce_stationarity=False,
                                    enforce_invertibility=False)
    results = mod.fit()
    print(results.summary())
    # 预测未来1年的数据
    forecast = results.get_forecast(steps=12)
    # 得到预测的置信区间
    forecast_ci = forecast.conf_int()
    y_forecast = forecast.predicted_mean
    y_pred_concat = pd.concat([y_forecast, forecast_ci], axis=1)
    y_pred_concat.columns = [u'预测值', u'下限', u'上限']
    data = [x, x, x, x, x, x, x, x, x, x, x, x]
    y_pred_concat.insert(0, '城市&产品类型', data)
    print(y_pred_concat)  # ARIMA(0,1,1)只能预测1期，之后的预测值与第一期相同
    df_forecast = df_forecast.append(y_pred_concat, ignore_index=False)
df_forecast.to_excel("预测结果.xlsx")
