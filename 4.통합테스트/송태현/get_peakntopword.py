from pandas_datareader import data, wb
import pandas as pd
import numpy as np
import datetime
import yfinance as yf # yahoo finance API    # pip install yfinance
import investpy # investing.com API          # pip install investpy
from pykrx import stock # krx API           # pip instasll pykrx
import talib as ta # 기술적 분석 (보조지표)
import FinanceDataReader as fdr
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
from datetime import datetime

import seaborn as sns

# 시각화 사용자 설정
from matplotlib import rcParams
sns.set_style('whitegrid')

from sklearn.preprocessing import MinMaxScaler
from scipy.signal import find_peaks
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional

from bs4 import BeautifulSoup
import requests

from selenium import webdriver
from selenium.webdriver.chrome.options import Options

from konlpy.tag import Okt
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter

def get_SAMSUNG_dataframe():
    start_date='2020-09-01'
    end_date='2021-03-01'
    # investing.com 양식, ex) dd/mm/yyyy
    start_date_ = start_date[8:] + '/' + start_date[5:7] + '/' + start_date[:4]
    end_date_ = end_date[8:] + '/' + end_date[5:7] + '/' + end_date[:4]

    # krx 양식 ex) yyyymmdd
    start_date__ = start_date[0:4] + start_date[5:7] + start_date[8:10]
    end_date__ = end_date[0:4] + end_date[5:7] + end_date[8:10]
    model_samsung = pd.DataFrame()

    # S&P 500
    snp_500_ = yf.download("^GSPC", start=start_date, end=end_date)
    snp_500_.columns = ['snp_500_Open','snp_500_High','snp_500_Low','snp_500_Close','snp_500_Adj Close','snp_500_Volume']
    snp_500_ = snp_500_.drop(['snp_500_Open','snp_500_High','snp_500_Low','snp_500_Adj Close','snp_500_Volume'], axis=1)

    # 삼성 차트 데이터
    sam_ = stock.get_market_ohlcv_by_date(start_date__, end_date__, "005930")
    sam_.columns = ['Open','High','Low','Close','Volume']
    model_samsung = sam_.copy()

    # SOX 지수 엔비디아(8.9%), 텍사스인스트루먼트(8.6%), 퀄컴(7.7%), 인텔(7.4%) 반도체장비기업 18.52%
    SOXX = yf.download("SOXX", start=start_date, end=end_date)
    SOXX.columns = ['SOXX_Open','SOXX_High','SOXX_Low','SOXX_Close','SOXX_Adj Close','SOXX_Volume']
    SOXX = SOXX.drop(['SOXX_Open','SOXX_High','SOXX_Low','SOXX_Adj Close','SOXX_Volume'], axis=1)
    model_samsung['SNP500'] = snp_500_
    model_samsung['SOXX'] = SOXX
    model_samsung = model_samsung.dropna() # 결측치가 있는 행 제거
    maxClose = model_samsung['Close'].max()
    maxSNP500 = model_samsung['SNP500'].max()
    maxSOXX = model_samsung['SOXX'].max()
    model_samsung["rev_Close"] = maxClose- model_samsung['Close']
    model_samsung["rev_SNP500"] = maxSNP500- model_samsung['SNP500']
    model_samsung["rev_SOXX"] = maxSOXX- model_samsung['SOXX']
    # 차트 비교분석을 위한 정규화

    min_max_scaler = MinMaxScaler()
    fitted = min_max_scaler.fit(model_samsung)
    output = min_max_scaler.transform(model_samsung)
    output = pd.DataFrame(output, columns=model_samsung.columns, index=list(model_samsung.index.values))
    return output, model_samsung

def get_SAMSUNG_Peaks(output):
    start_date = '2020-11-01'
    end_date   = '2021-02-01'
    Close_samsung= output['Close'][start_date:end_date]
    min_Close_samsung= output['rev_Close'][start_date:end_date]
    price_peak_limit = 0.1
    peaks, properties = find_peaks(Close_samsung, height=price_peak_limit)
    price_peak_limit = 0.1
    peaks2, properties2 = find_peaks(min_Close_samsung, height=price_peak_limit)
    max_peaks = peaks
    min_peaks = peaks2
    new_peaks = np.concatenate((peaks, peaks2),axis=0)
    new_peaks = np.sort(new_peaks)
    new_peaks
    plt.figure(figsize = (14, 8))
    plt.plot_date(Close_samsung.index, Close_samsung, 'k-', linewidth = 1)
    plt.plot_date(Close_samsung.index[peaks], Close_samsung[peaks], 'ro', label = 'samsung up')
    plt.plot_date(Close_samsung.index[peaks2], Close_samsung[peaks2], 'bo', label = 'samsung down')

    plt.xlabel('Datetime')
    plt.ylabel('Close')
    plt.legend(loc = 4)
    plt.show()
    
    Close_SNP500= output['SNP500'][start_date:end_date]
    Close_SOXX= output['SOXX'][start_date:end_date]
    min_SNP500= output['rev_SNP500'][start_date:end_date]
    min_SOXX= output['rev_SOXX'][start_date:end_date]
    price_peak_limit = 0.1
    peaks3, properties3 = find_peaks(Close_SNP500, height=price_peak_limit)
    peaks4, properties4 = find_peaks(Close_SOXX, height=price_peak_limit)

    peaks5, properties5 = find_peaks(min_SNP500, height=price_peak_limit)
    peaks6, properties6 = find_peaks(min_SOXX, height=price_peak_limit)
    plt.figure(figsize = (14, 8))
    plt.plot_date(Close_samsung.index, Close_samsung, 'k-', linewidth = 1)
    plt.plot_date(Close_SNP500.index, Close_SNP500, 'k-', linewidth = 1)
    plt.plot_date(Close_samsung.index[peaks], Close_samsung[peaks], 'ro', label = 'SAMSUNG UP')
    plt.plot_date(Close_SNP500.index[peaks3], Close_SNP500[peaks3], 'bo', label = 'SNP500 UP')

    for date in Close_samsung.index[peaks]:
        plt.axvline(x=date, color='g', linestyle='-', linewidth=1)
    plt.xlabel('Datetime')
    plt.ylabel('Close')
    plt.legend(loc = 4)
    plt.show()
    plt.figure(figsize = (14, 8))
    plt.plot_date(Close_samsung.index, Close_samsung, 'k-', linewidth = 1)
    plt.plot_date(Close_SNP500.index, Close_SNP500, 'k-', linewidth = 1)
    plt.plot_date(Close_samsung.index[peaks2], Close_samsung[peaks2], 'ro', label = 'SAMSUNG DOWN')
    plt.plot_date(Close_SNP500.index[peaks5], Close_SNP500[peaks5], 'bo', label = 'SNP500 DOWN')

    for date in Close_samsung.index[peaks2]:
        plt.axvline(x=date, color='g', linestyle='-', linewidth=1)
    plt.xlabel('Datetime')
    plt.ylabel('Close')
    plt.legend(loc = 4)
    plt.show()
    plt.figure(figsize = (14, 8))
    plt.plot_date(Close_samsung.index, Close_samsung, 'k-', linewidth = 1)
    plt.plot_date(Close_SOXX.index, Close_SOXX, 'k-', linewidth = 1)
    plt.plot_date(Close_samsung.index[peaks], Close_samsung[peaks], 'ro', label = 'SAMSUNG UP')
    plt.plot_date(Close_SOXX.index[peaks4], Close_SOXX[peaks4], 'bo', label = 'SOXX UP')

    for date in Close_samsung.index[peaks]:
        plt.axvline(x=date, color='g', linestyle='-', linewidth=1)
    plt.xlabel('Datetime')
    plt.ylabel('Close')
    plt.legend(loc = 4)
    plt.show()
    plt.figure(figsize = (14, 8))
    plt.plot_date(Close_samsung.index, Close_samsung, 'k-', linewidth = 1)
    plt.plot_date(Close_SOXX.index, Close_SOXX, 'k-', linewidth = 1)
    plt.plot_date(Close_samsung.index[peaks2], Close_samsung[peaks2], 'ro', label = 'SAMSUNG UP')
    plt.plot_date(Close_SOXX.index[peaks6], Close_SOXX[peaks6], 'bo', label = 'SOXX UP')

    for date in Close_samsung.index[peaks2]:
        plt.axvline(x=date, color='g', linestyle='-', linewidth=1)
    plt.xlabel('Datetime')
    plt.ylabel('Close')
    plt.legend(loc = 4)
    plt.show()    

def getPageNewsData(news_url, page) :
    url = news_url + "&page=" + str(page)
    response = requests.get(url)
    response.encoding = 'euc-kr' #한글깨짐방지
    soup = BeautifulSoup(response.text, 'html.parser')
    
    table = soup.find('table')
    rlist = table.find_all('tr', class_="relation_lst")
    for tr in rlist :
        tr.decompose()

    df = pd.read_html(str(table), header=0)[0].dropna()
    return df

def getLastPageNum(soup) :
    return int(soup.find("table", class_="Nnavi").find("td", class_="pgRR").find("a").get("href").split("&")[1].split("=")[1])

def get_news():
    code = '005930'
    news_url = 'https://finance.naver.com/item/news_news.nhn?code='+code
    startDate = '2020.09.01'
    print("네이버 기사를 가져옵니다.(약 5분 소요)")
    response = requests.get(news_url)
    soup = BeautifulSoup(response.text, 'html.parser')

    #마지막 페이지 가져오기
    last_page = getLastPageNum(soup)

    #페이지별로 loop 돌면서 dataframe으로 만들기
    last_index = last_page+1

    news_df = getPageNewsData(news_url, 1).dropna()
    for i in range(2, last_index) :
        tmp_df = getPageNewsData(news_url, i)
        add_df = tmp_df[tmp_df['날짜'] > startDate]
        news_df = pd.concat([news_df, add_df], ignore_index=True)
        if len(add_df) < len(tmp_df) : 
            break

    #추가 : 중복제거
    news_df = news_df.drop_duplicates()

    # 데이터셋으로 쓸 dataframe 만들기
    # 날짜 / 제목 / updown
    news_df['date'] = news_df['날짜'].str.split(' ').str[0]
    pivot_df = news_df.drop(['정보제공', '날짜'], axis=1).rename({'제목':'subject'}, axis='columns')
    return pivot_df

def clean(sub) :
    text = re.sub("[^ㄱ-ㅣ가-힣]", " ", sub)
    text2 = text.split()
    return text2

def get_top_words(pivot_df, model_samsung, output):
    start_date = '2020-09-01'
    end_date   = '2021-03-01'
    Close_samsung= output['Close'][start_date:end_date]
    min_Close_samsung= output['rev_Close'][start_date:end_date]
    price_peak_limit = 0.1
    peaks, properties = find_peaks(Close_samsung, height=price_peak_limit)
    price_peak_limit = 0.1
    peaks2, properties2 = find_peaks(min_Close_samsung, height=price_peak_limit)
    max_peaks = peaks
    min_peaks = peaks2

    pivot_df= pivot_df.reset_index()
    del pivot_df["index"]
    size = len(pivot_df)
    for s in range(size):
        pivot_df['date'][s] = pivot_df['date'][s][0:4]+pivot_df['date'][s][5:7]+pivot_df['date'][s][8:10]
    save_pivot_df = pivot_df
    model_samsung = model_samsung.reset_index()
    MAX = [] ## 최대치일때의 일자에 대한 뉴스 헤드라인을 넣는다.
    MIN = [] ## 최저치일때의 일자에 대한 뉴스 헤드라인을 넣는다.
    size = len(save_pivot_df)
    for day in max_peaks:
        for s in range(size):
            Sdate = model_samsung['날짜'][day].strftime('%Y%m%d')
            if(save_pivot_df['date'][s]== Sdate):
                MAX.append(save_pivot_df['subject'][s])

    for day in min_peaks:
        for s in range(size):
            Sdate = model_samsung['날짜'][day].strftime('%Y%m%d')
            if(save_pivot_df['date'][s]== Sdate):
                MIN.append(save_pivot_df['subject'][s])
    okt = Okt()
    sub_MAX_set = []
    sub_MIN_set = []

    for i in range(0, len(MAX)) :
        sub_MAX_set.append(" ".join(clean(MAX[i])))

    for i in range(0, len(MIN)) :
        sub_MIN_set.append(" ".join(clean(MIN[i])))

    sub_MAX_set_ml = [" ".join(okt.nouns(sub)) for sub in sub_MAX_set]
    sub_MAX_set_dl = [okt.nouns(sub) for sub in sub_MAX_set]
    sub_MIN_set_ml = [" ".join(okt.nouns(sub)) for sub in sub_MIN_set]
    sub_MIN_set_dl = [okt.nouns(sub) for sub in sub_MIN_set]
    vectorizer = CountVectorizer()
    MAX_data_features = vectorizer.fit_transform(sub_MAX_set_ml)
    np.asarray(MAX_data_features)
    MIN_data_features = vectorizer.fit_transform(sub_MIN_set_ml)
    np.asarray(MIN_data_features)
    tokens = []
    for news_tit in sub_MAX_set_ml:
        word = news_tit.split()
        tokens.extend(word)

    for a in tokens[::-1]:
        if (a.find('삼성') and a.find('전자') and a.find('만') and a.find('주') and a.find('위')and a.find('장')and a.find('명')
           and a.find('년') and a.find('특징')and a.find('총') and a.find('조') and a.find('원')) > -1:
            tokens.remove(a)

    counted_tokens = Counter(tokens)
    top_10 = counted_tokens.most_common(10)


    plt.rc('font', family='Malgun Gothic', size=15)
    plt.subplots(figsize=(12,4))
    plt.bar(*zip(*top_10))
    plt.show()
    print("최대치 PEAK에서 TOP10 WORDS")
    print(top_10)
    
    
    tokens = []
    for news_tit in sub_MIN_set_ml:
        word = news_tit.split()
        tokens.extend(word)

    for a in tokens[::-1]:
        if (a.find('삼성') and a.find('전자') and a.find('만') and a.find('주') and a.find('위')and a.find('장')and a.find('명')
           and a.find('년') and a.find('특징') and a.find('총') and a.find('조') and a.find('원')) > -1:
            tokens.remove(a)

    counted_tokens = Counter(tokens)
    top_10 = counted_tokens.most_common(10)

    plt.rc('font', family='Malgun Gothic', size=15)
    plt.subplots(figsize=(12,4))
    plt.bar(*zip(*top_10))
    plt.show()
    print("최저치 PEAK에서 TOP10 WORDS")
    print(top_10)
    
