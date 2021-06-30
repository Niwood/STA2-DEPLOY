import os
import pandas as pd
import pandas_ta as ta
from matplotlib.backend_bases import MouseButton
import matplotlib.pyplot as plt
from matplotlib.widgets import MultiCursor, RadioButtons, Button
import matplotlib.dates as mdates
import numpy as np
from datetime import datetime, timedelta
import pytz
import glob
from pathlib import Path
from tqdm import tqdm
import pickle
import random


import yfinance as yf

from scipy.stats import zscore
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.ensemble import IsolationForest

import tflite_runtime.interpreter as tflite





class TickUnit:


    def __init__(self, tick=None):

        # Parameters
        self.tick = tick
        self.net_name = '1624913277'
        self.num_steps = 90
        self.quantile_thr = 0.01 #for RSI threshold affirmation - lower means tighter
        self.now = datetime.now(pytz.utc)
        self.ticker = yf.Ticker(self.tick)
        self.last_notification_sent = datetime.now(pytz.utc)

        # Gain
        self.gain = Gain()

        # Paths
        self.rsi_diff_folder = Path.cwd() / 'data' / 'rsi_diff'
        self.net_path = Path.cwd() / 'networks' / f'{self.net_name}.tflite'

        # Assert the tick
        self._assert_tick()
        self._fetch()


    def infer(self):
        ''' Model inference '''
        self.last_infered = datetime.now(pytz.utc)
        self._load()
        self._fetch()
        self._data_process()
        trigger = self._predict()
        return trigger


    def backtest(self):
        ''' Backtestning '''
        pass


    def _assert_tick(self):
        ''' Assert that the tick is available '''
        print(f'Asserting {self.tick} ..')
        _info = self.ticker.info
        assert len(_info) > 1, f'Tick assertion failed: {self.tick} is not available'


    def _load(self):
        ''' LOAD NET MODEL '''
        self.net = tflite.Interpreter(model_path=str(self.net_path))
        self.net.allocate_tensors()

        self.input_details = self.net.get_input_details()[0]
        self.output_details = self.net.get_output_details()[0]

        ''' LOAD RSI DIFF '''
        with open(self.rsi_diff_folder / 'rsi_diff_sell.pkl', 'rb') as handle:
            self.rsi_diff_sell = pickle.load(handle)
        with open(self.rsi_diff_folder / 'rsi_diff_buy.pkl', 'rb') as handle:
            self.rsi_diff_buy = pickle.load(handle)

        # Determine rsi_diff quantile threshold
        self.rsi_diff_buy_thr = dict()
        self.rsi_diff_sell_thr = dict()
        for rsi_idx in self.rsi_diff_buy.keys():
            self.rsi_diff_buy_thr[rsi_idx] = np.quantile(self.rsi_diff_buy[rsi_idx], self.quantile_thr, axis=0) #generally negative
            self.rsi_diff_sell_thr[rsi_idx] = np.quantile(self.rsi_diff_sell[rsi_idx], 1-self.quantile_thr, axis=0) #generally positive


    def _fetch(self):
        ''' FETCH HIST DATA '''
        self.df = self.ticker.history(start=self.now-timedelta(hours=self.num_steps*10), end=self.now, interval='1h').tz_convert('UTC')
        self._fetch_now()


        self.df_org = self.df.copy()

        
    def _fetch_now(self):
        ''' FETCH LIVE DATA '''
        df_now = self.ticker.history(period='1d', interval='5m').tz_convert('UTC')
        df_now_whole_hours  = df_now[df_now.index.minute == 0].copy()


        # Append missing last hour
        self.df = self.df.append(df_now_whole_hours.iloc[-1])

        # Append latest reading
        self.df = self.df.append(df_now.iloc[-1])

        # Drop rows with duplicated indicies - same timestamp
        self.df = self.df[~self.df.index.duplicated(keep='first')]


    def get_last(self):
        return self.df_org.iloc[-1]



    def _isoforest(self, df):
        ''' Generate anomalie detection based in isolation forest '''
        
        df_org = df.copy()

        df['RSI_SMA'] = df.RSI_14.rolling(window=5).mean()
        df['RSI_SMA_diff'] = (df.RSI_14 - df.RSI_SMA)
        df.dropna(inplace=True)

        cols = ['RSI_14', 'RSI_SMA_diff']
        df = df[cols]
        
        # Isolation Forest prediction
        clf = IsolationForest(contamination=0.08, bootstrap=False, max_samples=0.99, n_estimators=200).fit(df)
        predictions = clf.predict(df) == -1

        # Insert anomalies
        df.insert(0, 'anomaly', predictions)
        df_org = df_org.join(df.anomaly)

        df_org.insert(0, 'peak_anomaly', (df_org.anomaly & (df_org.RSI_14.diff() > 0)))
        df_org.insert(0, 'valley_anomaly', (df_org.anomaly & (df_org.RSI_14.diff() < 0)))
        df_org.drop(['anomaly'], axis=1, inplace=True)

        return df_org


    def _data_process(self):
        ''' DATA PRE-PROCESSING '''

        # MACD
        self.df.ta.macd(fast=12, slow=26, append=True)

        # RSI
        self.df.ta.rsi(append=True)
        self.df.RSI_14 /= 100

        # Anomaly detection
        self.df = self._isoforest(self.df.copy())
    
        # BBAND - Bollinger band upper/lower signal - percentage of how close the hlc is to upper/lower bband
        bband_length = 30
        bband = self.df.ta.bbands(length=bband_length)
        bband['hlc'] = self.df.ta.hlc3()

        bbu_signal = (bband['hlc']-bband['BBM_'+str(bband_length)+'_2.0'])/(bband['BBU_'+str(bband_length)+'_2.0'] - bband['BBM_'+str(bband_length)+'_2.0'])
        bbl_signal = (bband['hlc']-bband['BBM_'+str(bband_length)+'_2.0'])/(bband['BBL_'+str(bband_length)+'_2.0'] - bband['BBM_'+str(bband_length)+'_2.0'])

        self.df['BBU_signal'] = bbu_signal
        self.df['BBL_signal'] = bbl_signal

        # Drop na and cut
        self.df.dropna(inplace=True)
        self.df = self.df.iloc[len(self.df)-self.num_steps::]

        # Assert for length
        assert len(self.df) == self.num_steps, f'Dataframe for {self.tick} has length {len(self.df)}, requires {self.num_steps}'

        # Reset and save index
        self.date_index = self.df.index.copy()
        self.df = self.df.reset_index(drop=False)

        # Choose columns for model input
        cols = ['MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9', 'RSI_14','BBU_signal', 'BBL_signal', 'peak_anomaly', 'valley_anomaly']
        self.df = self.df[cols]

        # Zscore and scale
        columns_to_scale = ['MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9' ,'BBU_signal', 'BBL_signal']
        self.df[columns_to_scale] = self.df[columns_to_scale].apply(zscore)
        scaler = MinMaxScaler()
        self.df[columns_to_scale] = scaler.fit_transform(self.df[columns_to_scale])

        # Round values
        self.df = self.df.round(decimals=5)

        # Assert for scaling
        for col in self.df.columns:
            assert self.df[col].max() <= 1.0 , f'In {self.tick}, max value in {col} is {self.df[col].max()}, requires <= 1'
            assert self.df[col].min() >= 0.0 , f'In {self.tick}, min value in {col} is {self.df[col].min()}, requires >= 0'



    def _predict(self):

        ''' Prediction algo '''
        current_price = round(self.df_org.Close.iloc[-1], 3)

        # Update gain
        self.gain.update(current_price)

        # Init trigger
        trigger = Trigger()
        trigger.reset()


        ''' Net prediction '''
        # Convert to numpy and reshape
        X = self.df.to_numpy()
        X = X.reshape((1 , *X.shape))

        # Edit format to float32
        X = np.asarray(X).astype('float32')

        # Set tensor to network, invoke and predict
        self.net.set_tensor(self.input_details['index'], X)
        self.net.invoke()
        prediction = self.net.get_tensor(self.output_details['index'])[0]
        action = np.argmax(prediction)
        trigger.set(f'Model predicts: {action}', action, override=False)

        # Model certainty threshold
        if trigger.action in (1,2):
            if max(prediction) < 0.90:
                trigger.set(f'Below model certainty threshold ({str(round(max(prediction),2))}): {action} -> 0', 0)


        ''' RSI threshold affirmation - if predicted hold '''
        if trigger.action == 0:
            for rsi_idx in self.rsi_diff_buy.keys(): #any of the buy/sell keys are ok since they are the same
                rsi_grad = (self.df.RSI_14.iloc[-1] - self.df.RSI_14.iloc[-int(rsi_idx)-1]) / rsi_idx

                if rsi_grad < self.rsi_diff_buy_thr[rsi_idx] and self.env.shares_held==0:
                    trigger.set(f'RSI threshold affirmation: Below {self.quantile_thr*100}% {rsi_idx} day(s) threshold -> buy', 1, override=False)

                elif rsi_grad > self.rsi_diff_sell_thr[rsi_idx] and self.env.shares_held>0:
                    trigger.set(f'RSI threshold affirmation: Above {(1-self.quantile_thr)*100}% {rsi_idx} day(s) threshold -> sell', 2, override=False)


        ''' MACD assertion '''
        if trigger.action==1 and self.df.MACDh_12_26_9.iloc[-1]>0:
            trigger.set(f'MACD assertion ({str(round(self.df.MACDh_12_26_9.iloc[-1],2))}): Failed on buy -> hold', 0)
        elif trigger.action==2 and self.df.MACDh_12_26_9.iloc[-1]<0:
            trigger.set(f'MACD assertion ({str(round(self.df.MACDh_12_26_9.iloc[-1],2))}): Failed on sell -> hold', 0)


        ''' --- COMMITED TO ACTION FROM THIS POINT --- '''

        if trigger.action == 1:
            self.gain.buy(current_price)

        elif trigger.action == 2:
            self.gain.sell()


        return trigger

        


class Trigger:
    def __init__(self):
        self.desc = None
        self.override = False
        self.action = None
        

    def set(self, description, action, override=False):
        if not self.override:
            self.desc = description
            self.override = override
            self.action = action

    def render(self):
        print('='*10)
        print(f'Action: {self.action}')
        print('='*10)

    def reset(self):
        self.desc = None
        self.override = False




class Gain:

    def __init__(self):
        self.buy_price = 0
        self.gain = 0
        self.gains = list()
        self.position = 'out'

    def update(self, price):
        if self.buy_price > 0:
            self.gain = (price / self.buy_price) - 1
        else:
            self.gain = 0
        self.gains.append(self.gain)

    def buy(self, buy_price):
        self.buy_price = buy_price
        self.position = 'in'

    def sell(self):
        self.buy_price = 0
        self.gain = 0
        self.position = 'out'



if __name__ == '__main__':
    tick = TickUnit(tick='VOLV-B.ST')
    tick.get_last()
    print(f'---> EOL: {__file__}')