import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

# 시각화 사용자 설정
from matplotlib import rcParams
sns.set_style('whitegrid')

import warnings
warnings.filterwarnings('ignore')

import FinanceDataReader as fdr
import yfinance as yf

import joblib

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from tensorflow.keras.utils import to_categorical

import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from xgboost import plot_importance as xgb_plot_importance
from lightgbm import plot_importance as lgb_plot_importance

from scipy import stats
import plotly.express as px
from sklearn.decomposition import PCA

from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input, Dense, LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LayerNormalization
from keras.regularizers import l2
from tensorflow.keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint

startYear = '2010'
start_date = startYear+'-01-01'
end_date = '2021-02-20'

border1 = '2020-04-01'
border2 = '2020-04-02'
val_border1 = '2020-09-01'
val_border2 = '2020-09-02'
end = '2021-02-19'

num_step = 5

def scale(df, method='minmax', ori_scaler='', makeDf=True) :
    if ori_scaler == '' :
        if method == 'minmax' :
            scaler = MinMaxScaler()
        elif method == 'standard' :
            scaler = StandardScaler()
        fitted = scaler.fit(df)
    else :
        scaler = ori_scaler
    output = scaler.transform(df)
    if makeDf :
        output = pd.DataFrame(output, columns=df.columns, index=list(df.index.values))
    return (scaler, output)

def machineLearning_Classifier(X_train, y_train, X_test, y_test) :
    neighbor_model = KNeighborsClassifier(n_neighbors=5)
    svm_model = SVC()
    forest_model = RandomForestClassifier(n_estimators=300)
    gbm_model = GradientBoostingClassifier(random_state=10)
    xgb_model = XGBClassifier(n_estimators=300, eval_metric = "logloss")
    lgb_model = LGBMClassifier(n_estimators=300)

    model_list = [neighbor_model, svm_model, forest_model, gbm_model, xgb_model, lgb_model]
    max_model_name = ''
    max_model = None
    max_model_acc = 0
    
    for model in model_list:
        model_name = model.__class__.__name__
        model.fit(X_train , y_train)
        print('\n{0} 학습데이터셋 정확도: {1:.4f}'.format(model_name, model.score(X_train , y_train)))

        pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, pred)    
        print('{0} 테스트셋 정확도: {1:.4f}'.format(model_name, accuracy)) 
        if accuracy > max_model_acc :
            max_model_acc = accuracy
            max_model_name = model_name
            max_model = model
        
    print(f"\n{max_model_name} 의 테스트 정확도가 제일 높으므로, 해당 모델을 저장합니다.")
    joblib.dump(max_model, 'ml_model.pkl')
    return model_list

def pcaConv(n_components, X, drawGraph=False, printMsg=False) : 

    columns = ['PCA_'+str(x) for x in range(0, n_components)]

    pca = PCA(n_components=n_components) 
    pca_fit = pca.fit(X)
    # pca 저장
    joblib.dump(pca_fit, 'pca_fit_classifier.pkl')
    printcipalComponents = pca_fit.transform(X)
    principalDf = pd.DataFrame(data=printcipalComponents, columns = columns, index=X.index)
    if printMsg :
        print(f'주성분 {n_components}개로 전체 데이터의 분산을 {str(sum(pca.explained_variance_ratio_)*100)}만큼 설명')

    if drawGraph :
        exp_var_cumul = np.cumsum(pca.explained_variance_ratio_)

        fig = px.area(
            x=range(1, exp_var_cumul.shape[0] + 1),
            y=exp_var_cumul,
            labels={"x": "# Components", "y": "Explained Variance"}
        )
        fig.show()
    return principalDf

def plot_history(history):
    
    plt.figure(figsize=(15, 5))
    ax = plt.subplot(2, 2, 1)
    plt.plot(history.history["loss"])
    plt.title("Train loss")
    ax = plt.subplot(2, 2, 2)
    plt.plot(history.history["val_loss"])
    plt.title("Test loss")
    
    ax = plt.subplot(2, 2, 3)
    plt.plot(history.history["accuracy"])
    plt.title("Train accuracy")
    ax = plt.subplot(2, 2, 4)
    plt.plot(history.history["val_accuracy"])
    plt.title("Test accuracy")
    plt.tight_layout()
    
    plt.savefig('sample.png')
    plt.show()
    return
    
def plot_roc(pred,y):
    fpr, tpr, _ = roc_curve(y, pred)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.show()
    return

def makeLSTMData(data, feature_list, step, n):
    train_xdata = np.array(data[feature_list[0:n]])
    train_ydata = np.array(data[[feature_list[n]]])

    m = np.arange(len(train_xdata) - step)

    x, y = [], []
    for i in m:
        a = train_xdata[i:(i+step)]
        x.append(a)

    x_batch = np.reshape(np.array(x), (len(m), step, n))

    for i in m + step - 1 :
        label = train_ydata[i][0]
        y.append(label)
    y_batch = np.reshape(np.array(y), (-1,1))

    return x_batch, y_batch

def makeFeatures() :
    print("학습 및 테스트 데이터를 생성하고 있습니다.")
    samsung = fdr.DataReader('005930', startYear)
    col = ['Open', 'High', 'Low', 'Volume', 'Change']
    date = ['2018-04-30', '2018-05-02', '2018-05-03']
    prevDate = '2018-04-27'

    for i in date :
        for j in col :
            samsung[i:i][j] = samsung[prevDate:prevDate][j].values[0]

    samsung[:'2018-05-03']['Volume'] = samsung[:'2018-05-03']['Volume'] * 50

    samsung['Open_Change'] = samsung['Open'].pct_change()
    samsung['High_Change'] = samsung['High'].pct_change()
    samsung['Low_Change'] = samsung['Low'].pct_change()

    vol = pd.DataFrame(index=samsung.index)
    vol_train = pd.DataFrame(index=samsung.index)
    vol_test = pd.DataFrame(index=samsung.index)

    vol_train['Close'] = samsung['Volume'][:border1]
    vol_test['Close'] = samsung['Volume'][border2:end]

    vol_scaler, vol_train = scale(vol_train)
    vol_test = scale(vol_test, ori_scaler=vol_scaler)[1]
    joblib.dump(vol_scaler, 'vol_scaler.pkl')

    vol_train.dropna(axis=0, inplace=True)
    vol_test.dropna(axis=0, inplace=True)
    vol = pd.concat([vol_train, vol_test])

    sox = fdr.DataReader('SOXX', startYear)
    usdkrw = fdr.DataReader('USD/KRW', startYear) 
    sp500 = yf.download("^GSPC", start=start_date, end=end_date)
    sp500['Change'] = sp500['Adj Close'].pct_change()
    sp500.dropna(inplace=True, axis=0)
    dji = fdr.DataReader('DJI', startYear)
    vix = yf.download("^VIX", start=start_date, end=end_date)
    vix['Change'] = vix['Adj Close'].pct_change()
    vix.dropna(inplace=True, axis=0)
    treasury_5y = yf.download("^FVX", start=start_date, end=end_date)
    treasury_5y['Change'] = treasury_5y['Adj Close'].pct_change()
    treasury_5y.dropna(inplace=True, axis=0)
    hangseng = fdr.DataReader('HSI', startYear)
    micron = fdr.DataReader('MU', startYear)

    feature_Change = pd.DataFrame(index=samsung.index)

    feature_Change['samsung'] = samsung.Change
    feature_Change['sox'] = sox.Change
    feature_Change['usdkrw'] = usdkrw.Change
    feature_Change['sp500'] = sp500.Change
    feature_Change['dji'] = dji.Change
    feature_Change['vix'] = vix.Change
    feature_Change['treasury_5y'] = treasury_5y.Change
    feature_Change['hangseng'] = hangseng.Change
    feature_Change['micron'] = micron.Change
    feature_Change['volume'] = vol.Close
    feature_Change['Open_Change'] = samsung.Open_Change
    feature_Change['High_Change'] = samsung.High_Change
    feature_Change['Low_Change'] = samsung.Low_Change

    feature_Change['Y_reg'] = feature_Change['samsung'].shift(-1)
    feature_Change['Y_bin'] = feature_Change.apply(lambda x : 1 if x['Y_reg'] > 0 else 0, axis=1)
    feature_Change = feature_Change['2012-05-09':end]
    feature_Change.fillna(method='ffill', inplace=True)

    X = feature_Change.drop(['Y_reg', 'Y_bin'], axis=1)
    Y = feature_Change['Y_bin']
    
    pca_df = pcaConv(8, X)
    pca_df['Y'] = Y
    print("데이터 생성이 완료되었습니다.")
    return pca_df

def makeMLModel(df) :
    print("\n머신러닝 모델을 만듭니다.\n")
    pca_df = df.copy()
    train_df  = pca_df.loc[:border1].copy()
    test_df   = pca_df.loc[border2:end].copy()

    start_train_date = pca_df.index[0].strftime("%Y-%m-%d")
    print(f"학습 기간은 {start_train_date} 부터 {border1} 까지입니다.")
    print(f"테스트 기간은 {border2} 부터 {end} 까지입니다.")
    
    x_train, y_train = train_df.drop(['Y'], axis=1), train_df['Y']
    x_test, y_test = test_df.drop(['Y'], axis=1), test_df['Y']

    model_list = machineLearning_Classifier(x_train, y_train, x_test, y_test)
    return

def makeDLModel(df) :
    print("\n딥러닝 모델을 만듭니다.")
    pca_df = df.copy()
    train_df  = pca_df.loc[:border1].copy()
    val_df = pca_df.loc[border2:val_border1].copy()
    test_df   = pca_df.loc[val_border2:].copy()

    start_train_date = pca_df.index[0].strftime("%Y-%m-%d")
    print(f"\n학습 기간은 {start_train_date} 부터 {border1} 까지입니다.")
    print(f"검증 기간은 {border2} 부터 {val_border1} 까지입니다.")
    print(f"테스트 기간은 {val_border2} 부터 {end} 까지입니다.")

    feature_list = pca_df.columns
    n_feature = len(feature_list)-1
    x_train, y_train = makeLSTMData(train_df[feature_list], feature_list, num_step, n_feature)
    x_val, y_val = makeLSTMData(val_df[feature_list], feature_list, num_step, n_feature)
    x_test, y_test = makeLSTMData(test_df[feature_list], feature_list, num_step, n_feature)

    y_train = to_categorical(y_train, 2)
    y_val = to_categorical(y_val, 2)
    y_test = to_categorical(y_test, 2)

    y_train_data = pd.DataFrame(y_train).sum()
    y_val_data = pd.DataFrame(y_val).sum()
    y_test_data = pd.DataFrame(y_test).sum()

    print("\nUp/Down 의 분포에 따른 정확도 기준값(최소 임계치)은 다음과 같습니다.")
    train_acc_base = round(max(y_train_data.values[0], y_train_data.values[1])/(y_train_data.values[0] + y_train_data.values[1]),4)
    val_acc_base = round(max(y_val_data.values[0], y_val_data.values[1])/(y_val_data.values[0] + y_val_data.values[1]),4)
    test_acc_base = round(max(y_test_data.values[0], y_test_data.values[1])/(y_test_data.values[0] + y_test_data.values[1]),4)
    print(f'train acc 기준값 : {train_acc_base}')
    print(f'val acc 기준값 : {val_acc_base}')
    print(f'test acc 기준값 : {test_acc_base}')

    print("\n학습을 진행하고 있습니다. 이 작업은 5분 정도 걸립니다.")
    t_batch_size = 60
    t_epochs = 100
    t_act1 = 'tanh'
    t_act2 = 'tanh'
    t_units1 = 504
    t_units2 = 504
    t_learning_rate = 0.0001
    t_dropOut1 = 0.2
    t_dropOut2 = 0

    model = keras.Sequential()
    model.add(LSTM(units=t_units1, 
               activation=t_act1, input_shape=(x_train.shape[1], x_train.shape[2]), 
               return_sequences = True, kernel_regularizer=l2(0.01)))
    model.add(Dropout(t_dropOut1))
    model.add(LayerNormalization())
    model.add(LSTM(units=t_units2, activation=t_act2, return_sequences = False))
    model.add(Dropout(t_dropOut2))
    model.add(Dense(2, activation='sigmoid'))
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=keras.optimizers.Adam(t_learning_rate))

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=10)
    mc = ModelCheckpoint('dl_model.h5', monitor='val_accuracy', mode='max', save_best_only=True)

    history = model.fit(x_train, y_train, epochs=t_epochs, batch_size=t_batch_size, validation_data=(x_val, y_val), callbacks=[es, mc], verbose=0)

    print("\n *** 학습 : epoch에 따른 loss 및 accuracy 변화 ***\n")
    plot_history(history)

    model = load_model('./dl_model.h5')
    predicted = model.predict(x_test)
    y_pred = np.argmax(predicted, axis=1)
    Y_test = np.argmax(y_test, axis=1)
    cm = confusion_matrix(Y_test, y_pred)
    report = classification_report(Y_test, y_pred)

    print("\n *** 테스트 데이터의 ROC 커브 ***\n")
    
    plot_roc(y_pred,Y_test)
    roc_score = roc_auc_score(Y_test,y_pred)
    print('ROC AUC 값 : {0:.4f}'.format(roc_score))

    tn = cm[0][0]
    fn = cm[1][0]
    tp = cm[1][1]
    fp = cm[0][1]
    if tp == 0:
        tp = 1
    if tn == 0:
        tn = 1
    if fp == 0:
        fp = 1
    if fn == 0:
        fn = 1

    TPR = round(float(tp)/(float(tp)+float(fn)), 2)
    FPR = round(float(fp)/(float(fp)+float(tn)), 2)
    accuracy = round((float(tp) + float(tn))/(float(tp) +
                                              float(fp) + float(fn) + float(tn)), 3)
    specitivity = round(float(tn)/(float(tn) + float(fp)), 3)
    sensitivity = round(float(tp)/(float(tp) + float(fn)), 3)

    add = round(float(fn)/(float(fn)+float(tp)), 2)

    print('\n======= 테스트 셋 정확도 측정 =======\n')
    print('실제 true 인 것 중 모델이 true 라고 예측한 비율: {}\n'.format(TPR))
    print('실제 false 인 것 중 모델이 false 라고 예측한 비율: {}\n'.format(specitivity))
    print('실제 false 인데 true 로 잘못 예측한 비율: {}\n'.format(FPR))
    print('실제 true 인데 false 로 잘못 예측한 비율 : {}\n'.format(add))

    print('accuracy (정확도): {}\n'.format(accuracy))
    return