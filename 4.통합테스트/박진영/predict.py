import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from datetime import datetime, date, timedelta

import warnings
warnings.filterwarnings('ignore')

import FinanceDataReader as fdr

# 시각화 사용자 설정
from matplotlib import rcParams
sns.set_style('whitegrid')
import matplotlib.colors as colors
from matplotlib.pyplot import cm

from sklearn.preprocessing import MinMaxScaler

from keras.models import load_model
import joblib
import talib
import yfinance as yf

from IPython.display import display, HTML

# 모델 저장 경로
filePath = './'

# LSTM time step
num_step = 5

# PCA 변환 차원
n_components = 8

# PCA 파일명
pcaPkl = 'pca_fit_classifier.pkl'

# 머신러닝 모델 파일명
ml_model = 'ml_model.pkl'

# 딥러닝 모델 파일명
dl_model = 'dl_model.h5'

# 강한 예측 확률 기준
prob_strength = 0.55

# volume scaler
vol_sclaer = 'vol_scaler.pkl'

nstate = [' 보유 ', '미보유']
tstate = ['매수', '매도', '유지']
updown = ['Up', 'Down']
pred_strength = ['강', '약']

def loadFeatureList(startDate, endDate, flag='backTesting') :
    print("*** 데이터 불러오는 중 ... ***")
    errMsg = "기준일에 해당하는 데이터가 없어 예측할 수 없습니다."
    
    startYear = str(int(startDate[0:4]) - 1)
    
    samsung = fdr.DataReader('005930', startYear)
    if flag == 'backTesting' :
        samsung['Y_close'] = samsung['Close'].shift(-1)
        samsung['Y'] = samsung['Change'].shift(-1)
    samsung['Open_Change'] = samsung['Open'].pct_change()
    samsung['High_Change'] = samsung['High'].pct_change()
    samsung['Low_Change'] = samsung['Low'].pct_change()
    samsung = samsung[:endDate]
    
    usdkrw = fdr.DataReader('USD/KRW', startYear, endDate)
    sox = fdr.DataReader('SOXX', startYear, endDate)
    dji = fdr.DataReader('DJI', startYear, endDate)
    hangseng = fdr.DataReader('HSI', startYear, endDate)
    micron = fdr.DataReader('MU', startYear, endDate)
    
    treasury_5y = yf.download("^FVX", start=startYear+'-01-01', end=endDate)
    sp500 = yf.download("^GSPC", startYear+'-01-01', end=endDate)
    vix = yf.download("^VIX", startYear+'-01-01', end=endDate)
    
    if any(x.empty for x in [treasury_5y, micron, hangseng, vix, dji, sp500, sox, usdkrw, samsung]) :
        print(errMsg)
        return None
    
    treasury_5y['Change'] = treasury_5y['Adj Close'].pct_change()
    sp500['Change'] = sp500['Adj Close'].pct_change()
    vix['Change'] = vix['Adj Close'].pct_change()
    
    vol = pd.DataFrame(index=samsung.index)
    vol['Close'] = samsung['Volume']
    
    # 기존의 scaler 로 scale
    vol_fit = joblib.load(filePath + vol_sclaer)
    vol_scaled = vol_fit.transform(vol)
    vol['Close'] = vol_scaled
    
    pivotLoc = samsung.index.get_loc(samsung[samsung.index == startDate].iloc[-1].name)
    
    compare_Change = pd.DataFrame(index = samsung.index)
    compare_Change['samsung'] = samsung.Change
    compare_Change['sox'] = sox.Change
    compare_Change['usdkrw'] = usdkrw.Change
    compare_Change['sp500'] = sp500.Change
    compare_Change['dji'] = dji.Change
    compare_Change['vix'] = vix.Change
    compare_Change['treasury_5y'] = treasury_5y.Change
    compare_Change['hangseng'] = hangseng.Change
    compare_Change['micron'] = micron.Change
    compare_Change['volume'] = vol.Close
    compare_Change['Open_Change'] = samsung.Open_Change
    compare_Change['High_Change'] = samsung.High_Change
    compare_Change['Low_Change'] = samsung.Low_Change
    
    if flag == 'backTesting' :
        compare_Change['Y'] = samsung.Y
        compare_Change['Close'] = samsung.Close
        compare_Change['Y_close'] = samsung.Y_close
        compare_Change['Y_close'] = samsung.Y_close.astype(int)
    
    if compare_Change.isnull().sum().sum() > 0 :
        print("※ 데이터 중 누락된 부분이 있어, 이전 영업일 데이터 값으로 대체합니다.\n")
        compare_Change.fillna(method='ffill', inplace=True)
        
    compare_Change = compare_Change.iloc[pivotLoc - num_step + 1 :]
    return compare_Change

def dataToPCA(data) :
    try :
        columns = ['PCA_'+str(x) for x in range(0, n_components)]
        pca_fit = joblib.load(filePath + pcaPkl)
        printcipalComponents = pca_fit.transform(data)
        principalDf = pd.DataFrame(data=printcipalComponents, columns = columns, index=data.index)
        return principalDf
    except :
        print("데이터 변환 시 문제가 발생하였습니다.")
        return None
    
def getDLModel(selectedModel = dl_model) :
    try :
        model = load_model(filePath + selectedModel)
        return model
    except :
        print("저장된 모델을 불러오는데 실패했습니다.")
        return None

def getMLModel(selectedModel = ml_model) :
    try :
        model = joblib.load(filePath + selectedModel)
        return model
    except :
        print("저장된 모델을 불러오는데 실패했습니다.")
        return None
    
def makeLSTMData(data):
    x_batch = np.reshape(np.array(data), (1, num_step, len(data.columns)))
    return x_batch

def predictReturn() :
    print("*** 삼성전자 익영업일 종가 UP/DOWN 예측에 따른 수익률 계산 ***\n")
    startDate = input("시작일을 입력하세요 EX) YYYY-MM-DD : \n")
    if startDate == "" :
        startDate='2021-02-19'
    endDate = input("종료일을 입력하세요 EX) YYYY-MM-DD : \n")
    if endDate == "" :
        endDate=(date.today() - timedelta(2)).isoformat()
    elif datetime.strptime(endDate,  "%Y-%m-%d") > (datetime.today() - timedelta(2)) :
        print("종료일을 조정합니다.")
        endDate=(date.today() - timedelta(2)).isoformat()
    predict_method = input("예측모델을 정해주세요 (ML / DL) : \n")
    if predict_method == "" :
        predict_method = "DL"
    else :
        predict_method = predict_method.upper()
    
    # 시작일, 종료일 영업일로 조정
    samsung = fdr.DataReader('005930', startDate, endDate)
    startDate = samsung.index[0].strftime("%Y-%m-%d")
    endDate = samsung.index[-1].strftime("%Y-%m-%d")
    print(f'시작일(영업일) : {startDate}')
    print(f'종료일(영업일) : {endDate}')
    print(f'예측모델 : {predict_method}\n')
    
    # 1. 대상기간동안의 feature data 불러오기
    data = loadFeatureList(startDate, endDate)
    
    if data is None :
        print("종료.")
        return
    
    # 2. 전제조건 출력
    print("\n※ 수익률 계산은 다음과 같은 가정 하에 수행됩니다.")
    print("  1. 당일 종가 대비 익일 종가 변화율을 예측합니다.")
    print("  2. 매매가 = 당일 종가 = 익일 시가라고 가정합니다.")
    print("  3. 매매 수수료와 세금은 고려하지 않습니다.")
    print("  4. 실제 주식 가격과 상관없이, 항상 현재 갖고있는 전체 자산을 투자한다고 가정합니다.")
    print("  5. 변화율이 + 일 때는 Up, 0 이거나 - 일때는 Down 으로 표시합니다.\n")
    
    # 3. 실행
    if predict_method == 'ML' :
        pd_columns = ['보유상태', '예측결과', '매매상태', '실제결과', '실제당일종가', '실제익일종가', '실제변화율', '잔고', '주식계좌', '수익률', '예측정확도']
    else :
        pd_columns = ['보유상태', '예측결과', '예측강도', '매매상태', '실제결과', '실제당일종가', '실제익일종가', '실제변화율', '잔고', '주식계좌', '수익률', '예측정확도']

    # 잔고
    bullet = 10000
    balance = bullet
    # 주식계좌
    purchase = 0
    # 보유상태 : 보유 / 미보유
    current_state = nstate[1]
    next_state = nstate[1]
    # 매매상태 : 매수 / 매도 / 유지
    trading_state = None
    # 정확도
    acc = None
    correct_num = 0
    iter_num = 1  
    
    if predict_method == 'ML' :
        data = data[startDate : endDate]
        load_model = getMLModel()
        if load_model is None :
            return
        iter_start = 0
        iter_end = len(data.index)
        init_log = pd.Series(['-', '-', '-', '-', '-', '-', '-', bullet, '-', '-', '-'], index=pd_columns)
    else :
        load_model = getDLModel()
        if load_model is None :
            return
        iter_start = num_step - 1
        iter_end = len(data.index)
        init_log = pd.Series(['-', '-', '-', '-', '-', '-', '-', '-', bullet, '-', '-', '-'], index=pd_columns)
   
    init_log = pd.DataFrame([init_log], columns=pd_columns, index=['시작'])
    
    result = pd.DataFrame(columns=pd_columns)
    result.index.name = '당일'
    result = pd.concat([result, init_log])
    
    for i in range(iter_start, iter_end):
        if predict_method == 'ML' :
            row = data.iloc[[i]]
        else :
            row = data.iloc[i-num_step+1:i+1]

        tmp_df = pd.DataFrame(row)
        tmp_df.drop(['Y', 'Close', 'Y_close'], inplace=True, axis=1)
        tmp_df = dataToPCA(tmp_df)
        if tmp_df is None :
            return None
        if predict_method == 'DL' :
            tmp_df = makeLSTMData(tmp_df)
            
        real_val = row['Y'].values[-1]
        real_val_percent = round(real_val * 100, 1)
        if real_val_percent > 0 :
            real_val_percent = '+' + str(real_val_percent)
            
        if real_val > 0 :
            real = updown[0]
        else :
            real = updown[1]

        tmp_df = tmp_df[-1:]
        pred = load_model.predict(tmp_df)
        if predict_method == 'DL' :
            softmax = tf.keras.layers.Softmax()
            prob = softmax(pred)[0].numpy()
            pred = np.argmax(pred, axis=1)[0]
            if prob[pred] > prob_strength :
                strength = pred_strength[0]
            else :
                strength = pred_strength[1]
        
        if pred == 1 :
            pred = updown[0]
        else :
            pred = updown[1]

        if real == pred :
            correct_num = correct_num + 1
        # 정확도 계산
        acc = round(correct_num / iter_num * 100, 2)
        
        # 현재 상태에 따른 다음 상태와 잔고, 주식계좌 계산
        current_state = next_state
        if pred == updown[0] :
            if current_state == nstate[1] :
                trading_state = tstate[0] 
                next_state = nstate[0]
                purchase = int(balance * (1 + real_val))
                balance = 0
            elif current_state == nstate[0] :
                trading_state = tstate[2]
                purchase = int(purchase * (1 + real_val))
        elif pred == updown[1] :
            if current_state == nstate[1] :
                trading_state = tstate[2]
            elif current_state == nstate[0] :
                trading_state = tstate[1]
                next_state = nstate[1]
                balance = purchase
                purchase = 0

        if balance == 0 and purchase != 0 :
            returnRate = round((purchase - bullet)/ bullet  * 100, 2)
            if returnRate > 0 :
                returnRate = '+' + str(returnRate) + '%'
            else :
                returnRate = str(returnRate) + '%'
        else :
            returnRate = round((balance - bullet)/ bullet  * 100, 2)
            if returnRate > 0 :
                returnRate = '+' + str(returnRate) + '%'
            else :
                returnRate = str(returnRate) + '%'
    
        if predict_method == 'DL' :
            trading_log = pd.Series([current_state, pred, strength, trading_state, real, row['Close'].values[-1], row['Y_close'].values[-1], str(real_val_percent)+'%', balance, purchase, returnRate, str(acc)+'%'], index=pd_columns)
        else :
            trading_log = pd.Series([current_state, pred, trading_state, real, row['Close'].values[-1], row['Y_close'].values[-1], str(real_val_percent)+'%', balance, purchase, returnRate, str(acc)+'%'], index=pd_columns)

        trading_log = pd.DataFrame([trading_log], columns=pd_columns, index=[row.index[-1].strftime(format='%Y-%m-%d')])
        result = pd.concat([result, trading_log])
        
        iter_num = iter_num + 1
    
    # 최종 수익률
    if balance == 0 and purchase != 0 :
            returnRate = str(round((purchase - bullet)/ bullet  * 100, 2)) + '%'
    else :
            returnRate = str(round((balance - bullet)/ bullet  * 100, 2)) + '%'
    
    # 바이앤홀드 전략과 비교
    samsung = fdr.DataReader('005930', startDate)
    endiloc = samsung.index.get_loc(samsung[samsung.index == endDate].iloc[-1].name)
    buyPrice = samsung['Close'][startDate]
    sellPrice = samsung['Close'].iloc[endiloc+1]
    buyNhold = round((sellPrice - buyPrice) / buyPrice * 100, 2)
    
    return (buyNhold, result, returnRate)

def tradingGraph(ori_df, col_price, col_trading, col_rtn) :
    df = ori_df.copy()
    
    df = df.iloc[1:]
    df[col_price] = df[col_price].astype(int)
    df.index = pd.to_datetime(df.index, format='%Y-%m-%d')
    buy_point = df[df[col_trading]==tstate[0]].index
    sell_point = df[df[col_trading]==tstate[1]].index
    df[col_rtn] = df[col_rtn].shift(1)
    
    df[col_rtn] = df[col_rtn].str.split('%').str[0]
    df[col_rtn] = df[[col_rtn]].applymap(lambda x: x if x != '-' else np.nan)
    df[col_rtn] = df[col_rtn].astype(float)
    up_mask = df[col_rtn] > 0
    down_mask = df[col_rtn] < 0
    
    plt.style.use('default')
    plt.rcParams['figure.figsize'] = 10,4
    plt.rcParams['font.size'] = 8
    plt.rc('font', family='NanumGothic')
    plt.rc('axes', unicode_minus=False) # 마이너스 폰트 설정
     
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
        
    ax2._get_patches_for_fill.prop_cycler = ax1._get_patches_for_fill.prop_cycler
    next(ax1._get_patches_for_fill.prop_cycler)
    next(ax1._get_patches_for_fill.prop_cycler)
    
    ax1.plot(df.index, df[col_price], label='종가', zorder=2)
    ax1.scatter(buy_point, df.loc[buy_point][col_price].to_list(), s=15, label=tstate[0], zorder=3)
    ax1.scatter(sell_point, df.loc[sell_point][col_price].to_list(), s=15, label=tstate[1], zorder=3)
    ax1.set_xlabel('일자')
    ax1.set_ylabel('주가(원)')

    next(ax1._get_patches_for_fill.prop_cycler)
    next(ax1._get_patches_for_fill.prop_cycler)
    
    ax2.bar(df.index[up_mask], df[col_rtn][up_mask], label='수익률(+)')
    ax2.bar(df.index[down_mask], df[col_rtn][down_mask], label='수익률(-)')
    ax2.set_ylabel('비율(%)')
    ax2.axhline(y=0, color='gray', linewidth=1)
    
    #for tick in ax1.get_xticklabels():
    #    tick.set_rotation(45)
    
    ax1.set_zorder(ax2.get_zorder() + 10)
    ax1.patch.set_visible(False)

    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
   
    return

def backTesting() :
    result = predictReturn()
    print(f"Buy & Hold 했을 경우 수익률 : {result[0]}%")
    print(f"예측 최종 수익률 : {result[2]}")
    display(result[1])
    tradingGraph(result[1], '실제당일종가', '매매상태', '수익률')
    return

def predictNext() :
    # 오늘이 영업일이라면, 15시 이전 / 이후로 나눈다
    # 영업일이 아니라면, 다음 영업일 값을 예측한다
    startDate = (date.today() - timedelta(10)).isoformat()
    samsung = fdr.DataReader('005930', startDate)
    if samsung.index[-1].strftime("%Y-%m-%d") == date.today().isoformat() :
        now = datetime.now()
        if now.hour < 15 :
            prevDate = samsung.index[-2]
        else :
            prevDate = samsung.index[-1]
    else :
        prevDate = samsung.index[-1]
    prevDate_str = prevDate.strftime("%Y-%m-%d")
    print(f"*** {prevDate_str} 일자를 기준으로 다음 영업일의 종가 변화율을 예측합니다. ***\n")

    data = loadFeatureList(prevDate_str, prevDate_str, "predictNext")

    if data is None :
        return

    ml_data = data[prevDate_str:].copy()
    ml_data = dataToPCA(ml_data)
    ml_model = getMLModel()
    ml_pred = ml_model.predict(ml_data)
    if ml_pred == 1 :
        ml_pred = updown[0]
    else :
        ml_pred = updown[1]
    print(f"머신러닝 모델로 예측한 결과는 {ml_pred} 입니다.\n")

    dl_data = data.copy()
    dl_data = dataToPCA(dl_data)
    dl_data = makeLSTMData(dl_data)
    dl_model = getDLModel()
    dl_pred = dl_model.predict(dl_data)
    softmax = tf.keras.layers.Softmax()
    prob = softmax(dl_pred)[0].numpy()
    dl_pred = np.argmax(dl_pred, axis=1)[0]
    if prob[dl_pred] > prob_strength :
        strength = pred_strength[0]
    else :
        strength = pred_strength[1]
    if dl_pred == 1 :
        dl_pred = updown[0]
    else :
        dl_pred = updown[1]
    print(f"딥러닝 모델로 예측한 결과는 {dl_pred}({strength}한 예측) 입니다.\n")
    return

def readMe() :
    print("*** 아래 순서로 실행합니다. ***\n")
    print("1. 데이터 불러오기 : data = makeFeatures()")
    print("2. 머신러닝 모델 만들기 : makeMLModel(data)")
    print("3. 딥러닝 모델 만들기 : makeDLModel(data)")
    print("4. 만든 모델로 백테스팅 해보기 : backTesting()")
    print("5. 만든 모델로 Up/Down 예측하기 : predictNext()")
    return