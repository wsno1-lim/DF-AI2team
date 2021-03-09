import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#!conda install -c conda-forge scikit-plot
#!pip install yfinance
#!pip install investpy
#!pip install pykrx
#!pip install seaborn
#!pip install workalendar
#!pip install --upgrade finance-datareader
#!pip install pandas_datareader
#!pip install -U finance-datareader
from pandas_datareader import data, wb
import pandas as pd
import numpy as np
import datetime
import yfinance as yf # yahoo finance API    # pip install yfinance
import investpy # investing.com API          # pip install investpy
from pykrx import stock # krx API           # pip instasll pykrx
import FinanceDataReader as fdr
import matplotlib.pyplot as plt
#%matplotlib inline
import requests as re
from bs4 import BeautifulSoup
from pandas import DataFrame, Series
import numpy as np
import pandas as pd
import os
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
folder_adress = 'C:/Users/woori/Downloads/model'


def get_trading_trend():
    url = 'http://finance.naver.com/sise/investorDealTrendDay.nhn?bizdate=20210217&sosok=&page='
    headers = {'User-Agent':'Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.132 Safari/537.36'}
    #response = requests.get(news_url,headers = headers)

    date_list = []
    private = []
    foreign = []
    institution = []

    for i in range(1,500):
        url_ = re.get(url + str(i),headers = headers)
        url_.encoding = 'euc-kr' #한글깨짐방지
        url_ = url_.content
        html = BeautifulSoup(url_,'html.parser')

        body = html.find('body')

        tr = body.find_all('tr')

        for r in tr:
            date = r.find('td',{'class':'date2'})

            if date != None:
                date = date.text.strip().replace('.','-')
                date = date[6:] + '-' + date[3:5] + '-' + date[:2]
                #date = date.text.insert('20-01-01')
                #date = date.text.insert('01-01-20')


                # 날짜가 중첩되지 않으면 계속 크롤링 :
                if not date in date_list:
                    date_list.append(date)
                    #print(date)

                # 더 이상 자료가 없어, 날짜가 중첩되면 : 크롤링 완료
                else:
                    #Data = DataFrame(index = date_list)
                    Data = DataFrame()
                    Data['date'] = date_list
                    Data['private'] = private
                    Data['foreign'] = foreign
                    Data['institution'] = institution

                    Data.to_csv(folder_adress+'/trading_trend.csv')
                    return Data

                td = r.find_all('td')

                count = 0

                # 앞에서 3개 값 '개인' , '외국인' , '기관' 만 가져온다
                for d in td:
                    if count != 3:
                        d = d.text.replace(',','')
                    try:
                        d = int(d)

                        if count == 0 :
                            private.append(d)
                        elif count == 1 :
                            foreign.append(d)
                        else:
                            institution.append(d)

                        count += 1

                    except:
                        count = count


def get_trading_trend_fix() :
    samsung_etc = pd.read_csv('C:/Users/woori/Downloads/model/trading_trend.csv',parse_dates=['date'])
    samsung_etc = samsung_etc.drop(['Unnamed: 0'], axis=1)
    samsung_etc = samsung_etc.set_index("date")
    return samsung_etc

def get_feature_info_get(samsung_etc) :
    # 크롤링 start_date, end_date
    start_date='2015-01-01'
    #input('YYYY-MM-DD 형식을 지켜 입력해주세요 ex) 2018-01-01 : ')
    end_date='2021-02-20'
    #input('YYYY-MM-DD 형식을 지켜 입력해주세요 ex) 2020-10-13 : ')
    # investing.com 양식, ex) dd/mm/yyyy
    start_date_ = start_date[8:] + '/' + start_date[5:7] + '/' + start_date[:4]
    end_date_ = end_date[8:] + '/' + end_date[5:7] + '/' + end_date[:4]

    # krx 양식 ex) yyyymmdd
    start_date__ = start_date[0:4] + start_date[5:7] + start_date[8:10]
    end_date__ = end_date[0:4] + end_date[5:7] + end_date[8:10]
    # 코로나 이전날짜
    # 주요 3개국 대비 원 환율
    # 달러/원
    exchange_rate_usd_ = investpy.get_currency_cross_historical_data(currency_cross='USD/KRW', from_date=start_date_, to_date=end_date_)
    exchange_rate_usd_.columns = ['exchange_rate_usd_Open', 'exchange_rate_usd_High', 'exchange_rate_usd_Low', 'exchange_rate_usd_Close', 'exchange_rate_usd_Currency']
    exchange_rate_usd_ = exchange_rate_usd_.drop(['exchange_rate_usd_Open','exchange_rate_usd_High','exchange_rate_usd_Low','exchange_rate_usd_Currency'], axis=1)

    # 삼성 차트 데이터
    sam_ = stock.get_market_ohlcv_by_date(start_date__, end_date__, "005930")
    sam_.columns = ['Open','High','Low','Close','Volume']
    model_samsung = sam_.copy()

    # 미국 국채 수익률 (5년)
    treasury_5y_ = yf.download("^FVX", start=start_date, end=end_date)
    treasury_5y_.columns = ['treasury_5y_Open','treasury_5y_High','treasury_5y_Low','treasury_5y_Close','treasury_5y_Adj Close','treasury_5y_Volume']
    treasury_5y_ = treasury_5y_.drop(['treasury_5y_Open','treasury_5y_High','treasury_5y_Low','treasury_5y_Adj Close','treasury_5y_Volume'], axis=1)

    # HANG SENG
    hang_seng_ = yf.download("^HSI", start=start_date, end=end_date)
    hang_seng_.columns = ['hang_seng_Open','hang_seng_High','hang_seng_Low','hang_seng_Close','hang_seng_Adj Close','hang_seng_Volume']
    hang_seng_ = hang_seng_.drop(['hang_seng_Open','hang_seng_High','hang_seng_Low','hang_seng_Close','hang_seng_Volume'], axis=1)

    #QQQ
    qqq=yf.download("QQQ", start=start_date, end=end_date)
    qqq.columns = ['qqq_Open','qqq_High','qqq_Low','qqq_Close','qqq_Adj Close','qqq_Volume']
    qqq = qqq.drop(['qqq_Open','qqq_High','qqq_Low','qqq_Close','qqq_Volume'], axis=1)

    # Russell 2000
    russell_2000_ = yf.download("^RUT", start=start_date, end=end_date)
    russell_2000_.columns = ['russell_2000_Open','russell_2000_High','russell_2000_Low','russell_2000_Close','russell_2000_Adj Close','russell_2000_Volume']
    russell_2000_ = russell_2000_.drop(['russell_2000_Open','russell_2000_High','russell_2000_Low','russell_2000_Adj Close','russell_2000_Volume'], axis=1)

    # SOX 지수 엔비디아(8.9%), 텍사스인스트루먼트(8.6%), 퀄컴(7.7%), 인텔(7.4%) 반도체장비기업 18.52%
    SOXX = yf.download("SOXX", start=start_date, end=end_date)
    SOXX.columns = ['SOXX_Open','SOXX_High','SOXX_Low','SOXX_Close','SOXX_Adj Close','SOXX_Volume']
    SOXX = SOXX.drop(['SOXX_Open','SOXX_High','SOXX_Low','SOXX_Close','SOXX_Volume'], axis=1)

    # S&P 500
    snp_500_ = yf.download("^GSPC", start=start_date, end=end_date)
    snp_500_.columns = ['snp_500_Open','snp_500_High','snp_500_Low','snp_500_Close','snp_500_Adj Close','snp_500_Volume']
    snp_500_ = snp_500_.drop(['snp_500_Open','snp_500_High','snp_500_Low','snp_500_Adj Close','snp_500_Volume'], axis=1)

    model_samsung = model_samsung.drop(['Open','High','Low'],axis=1)
    model_samsung[:'2018-05-03']['Volume'] = model_samsung[:'2018-05-03']['Volume'] * 50
    model_samsung = pd.concat([model_samsung, samsung_etc],axis=1)
    model_samsung['US'] = exchange_rate_usd_
    model_samsung['SNP500'] = snp_500_
    model_samsung['SOXX'] = SOXX
    model_samsung['hang_seng'] = hang_seng_
    model_samsung['qqq'] = qqq

    vix = fdr.DataReader('VIX', '1990-01-01', '2021-02-23') # S&P 500 VIX
    vix.iloc[-5:]
    vix.columns = ['vix', 'vix_open','vix_high','vix_low','vix_volume','vix_change']
    model_vix = vix['vix']
    model_samsung = pd.concat([model_samsung, model_vix],axis=1)
    model_samsung = model_samsung.dropna()

    return model_samsung

def get_shift_feature(model_samsung):
    new_model_samsung = pd.DataFrame(index=model_samsung.index)
    for list in model_samsung.columns:
        if list == 'Close' :
            new_model_samsung['Close'] = model_samsung[list]
        else :
            new_model_samsung[list+'_1D_shift'] = model_samsung[list].shift(-1).fillna(model_samsung.mean())
    new_model_samsung = new_model_samsung.dropna()
    return new_model_samsung

def get_model_scaler(model_samsung):
    min_max_scaler = MinMaxScaler()
    fitted = min_max_scaler.fit(model_samsung)
    output = min_max_scaler.transform(model_samsung)
    output = pd.DataFrame(output, columns=model_samsung.columns, index=list(model_samsung.index.values))
    return output


def get_model_heatmap(output):
    plt.figure(figsize=(8,8))
    sns.heatmap(output.corr(), annot=True, cmap='summer')

def get_graph(output) :
    for list in output.columns :
        if list != 'Close' :
            figure, ((ax1)) = plt.subplots(nrows=1, ncols=1, figsize=(20, 8))
            sns.lineplot(data=output[['Close', list]], ax=ax1)
            #ax1.axvspan('2007-01', '2008-12', alpha=0.3, color='red')
            #ax1.annotate('Financial crisis', xy=('2008', 70), fontsize=20)
            #ax1.axvspan('2019-12', '2020-12', alpha=0.3, color='red')
            #ax1.annotate('COVID-19 Pandemic', xy=('2019-12', 70), fontsize=20)
            plt.title('Samsung Close feature '+ list, fontsize=20)
            plt.ylabel('Value', fontsize=14)
            plt.xlabel('Date', fontsize=14)
            plt.legend(fontsize=12, loc='best')
