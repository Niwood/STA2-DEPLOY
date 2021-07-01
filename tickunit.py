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
from pytz import timezone
import glob
from pathlib import Path
from tqdm import tqdm
import pickle
import random
import json
import time


import yfinance as yf
import pandas_market_calendars as mcal

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
        self.last_infered = None

        # Market flags
        self.market_open = None
        self.next_market_open = None

        # Meta data
        self.compnay_name = None
        self.exchange = None
        self.exchange_timezone = None
        self.sector = None
        self.currency = None

        # Trigger
        self.trigger = Trigger()

        # Gain
        self.gain = Gain()

        # Paths
        self.config_path = Path.cwd() / 'config_files'
        self.rsi_diff_folder = Path.cwd() / 'data' / 'rsi_diff'
        self.net_path = Path.cwd() / 'networks' / f'{self.net_name}.tflite'

        # Assert the tick
        self._assert_tick()

        # Load config file if available
        self._load_config()

        # Make an initial fetch
        self._fetch()

        # Load network nad RSI diff
        self._load()


    def infer(self):
        ''' Model inference '''

        # Update last infered
        self.last_infered = datetime.now(pytz.utc)

        # Fetch data and process
        self._fetch()
        self._data_process()

        # Predict
        self._predict()

        # Save config file - has to be after predict
        self._save_config()


    def _assert_tick(self):
        ''' Assert that the tick is available and save meta data '''

        print(f'Asserting {self.tick} ..')
        company_info = self.ticker.info
        
        assert len(company_info) > 1, f'Tick assertion failed: {self.tick} is not available'

        self.compnay_name = company_info['shortName']
        self.exchange = company_info['exchange']
        self.exchange_timezone = company_info['exchangeTimezoneName']
        self.sector = company_info['sector']
        self.currency = company_info['currency']
        time.sleep(1)


    def _load_config(self):
        ''' Load config file '''

        try:
            with open(self.config_path / f'{self.tick}.json') as json_file:
                data = json.load(json_file)
        except:
            return

        self.gain.buy_price = data['buy_price']
        self.gain.gain = data['gain']
        self.gain.gains = data['gains']
        self.gain.position = data['position']


    def _save_config(self):
        ''' Save config file '''

        data = {
            'buy_price': self.gain.buy_price,
            'gain': self.gain.gain,
            'gains': self.gain.gains,
            'position': self.gain.position
        }

        with open(self.config_path / f'{self.tick}.json', 'w') as outfile:
            json.dump(data, outfile)


    def check_market_open(self):
        ''' Determines if the market is open, if not when will it open '''

        # Get timezone of the exchange
        tz_exchange = timezone(self.exchange_timezone)

        # Set market calendar - (X+STO)
        market_calendar = mcal.get_calendar('X'+self.exchange)

        # Get the schedule
        schedule = market_calendar.schedule(
            start_date=datetime.now(tz_exchange).date(),
            end_date=(datetime.now(tz_exchange)+timedelta(days=10)).date()
            )

        # Is the market is open this date
        try:
            schedule_today = schedule.loc[
                str(datetime.now(pytz.utc).date())
                ]
            _is_market_open_today = True
        except:
            _is_market_open_today = False

        # Is the market is open at this time
        if _is_market_open_today:
            market_open = schedule_today.market_open < datetime.now(pytz.utc) < schedule_today.market_close
        else:
            market_open = False
        
        # When will be the next time the market opens
        if _is_market_open_today:
            schedule_next_day = schedule.iloc[1]
        else:
            schedule_next_day = schedule.iloc[0]

        next_market_open = schedule_next_day.market_open

        # print('next_market_open: ',self.next_market_open)
        # print('market open:',self.market_open)
        # quit()
        return market_open, next_market_open


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
        fetched = False
        while not fetched:
            try:
                self.df = self.ticker.history(start=self.now-timedelta(hours=self.num_steps*10), end=self.now, interval='1h').tz_convert('UTC')
                self._fetch_now()
                fetched = True
            except:
                print(f'Not able to fetch {self.tick}, will retry in 5 seconds')
                time.sleep(5)
        
        # Save original
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


    def get_daily_return(self):
        df_today = self.df_org.loc[str(datetime.now(pytz.utc).date())]
        if not df_today.empty:
            _daily_return = (df_today.Close.iloc[-1] / df_today.Close.iloc[0]) - 1
            return _daily_return
        else:
            return 0


    def _isoforest(self, df):
        ''' Generate anomalie detection based in isolation forest '''
        
        df_org = df.copy()

        df['RSI_SMA'] = df.RSI_14.rolling(window=5).mean()
        df['RSI_SMA_diff'] = (df.RSI_14 - df.RSI_SMA)
        df.dropna(inplace=True)

        cols = ['RSI_14', 'RSI_SMA_diff']
        df = df[cols]
        
        # Isolation Forest prediction
        clf = IsolationForest(contamination=0.3, bootstrap=False, max_samples=0.99, n_estimators=200).fit(df)
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

        # Reset trigger
        self.trigger.reset()


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
        self.trigger.set(f'Net: {action}', action, override=False)

        # Model certainty threshold
        if self.trigger.action in (1,2):
            if max(prediction) < 0.90:
                self.trigger.set(f'Net thrs ({str(round(max(prediction),2))}): {action} -> 0', 0)


        ''' MACD assertion '''
        if self.trigger.action==1 and self.df.MACDh_12_26_9.iloc[-1]>0:
            self.trigger.set(f'MACDh ({str(round(self.df.MACDh_12_26_9.iloc[-1],2))}): buy -> hold', 0)
        elif self.trigger.action==2 and self.df.MACDh_12_26_9.iloc[-1]<0:
            self.trigger.set(f'MACDh ({str(round(self.df.MACDh_12_26_9.iloc[-1],2))}): sell -> hold', 0)


        ''' Stop loss assertion - only if a buy action has been performed '''
        if self.trigger.action != 2 and self.gain.position=='in' and self.gain.gain>=0.03:
            self.trigger.set(f'InvStopLoss ({str(round(self.gain.gain,3))}): {self.trigger.action} -> 2', 2)
        elif self.trigger.action != 2 and self.gain.position=='in' and self.gain.gain<-0.02:
            self.trigger.set(f'StopLoss ({str(round(self.gain.gain,3))}): {self.trigger.action} -> 2', 2)


        ''' Compare trigger action to current position '''
        if self.trigger.action == 1 and self.gain.position == 'in':
            self.trigger.set(f'Trigger {self.trigger.action} redundant --> hold', 0)
        elif self.trigger.action == 2 and self.gain.position == 'out':
            self.trigger.set(f'Trigger {self.trigger.action} redundant --> hold', 0)


        ''' --- COMMITED TO ACTION FROM THIS POINT --- '''

        if self.trigger.action == 1:
            self.gain.buy(current_price)

        elif self.trigger.action == 2:
            self.gain.sell()




class Trigger:

    def __init__(self):
        self.desc = None
        self.override = False
        self.action = None
        self.set_action_desc()
        
    def set(self, description, action, override=False):
        if not self.override:
            self.desc = description
            self.override = override
            self.action = action
            self.set_action_desc()

    def reset(self):
        self.desc = None
        self.override = False
        self.set_action_desc()

    def set_action_desc(self):
        if self.action == 1:
            self.action_desc = 'buy'
        elif self.action == 2:
            self.action_desc = 'sell'
        else:
            self.action_desc = 'hold'


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
    tick = TickUnit(tick='ERIC-B.ST')
    tick.get_daily_return()
    # tick.infer()
    print(f'---> EOL: {__file__}')