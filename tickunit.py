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
import logging

import yfinance as yf
import pandas_market_calendars as mcal

from utils.feature_engineer import FeatureEngineer

from scipy.stats import zscore
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.ensemble import IsolationForest

import tflite_runtime.interpreter as tflite





class TickUnit:


    def __init__(self, tick=None, net=None):

        # Parameters
        self.tick = tick
        self.net_name = '1630867357_lstmRegul1e-3_DSmax_lstmUnits1024_convFilters1024_lre-7'
        self.num_steps = 30
        self.now = datetime.now(pytz.utc)
        self.ticker = yf.Ticker(self.tick)
        self.last_notification_sent = datetime.now(pytz.utc)
        self.last_infered = None

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

        # Feature engineer
        self.feature_engineer = FeatureEngineer()

        # Paths
        self.config_path = Path.cwd() / 'config_files'

        # Assert the tick
        self._assert_tick()

        # Load config file if available
        self._load_config()

        # # Make an initial fetch
        # self._fetch(period='1mo')

        # Load network shapes
        self.net = net
        self.input_details = self.net.get_input_details()[0]
        self.output_details = self.net.get_output_details()[0]


    def infer(self):
        ''' Model inference '''
        logging.info(f'Performing inference on {self.tick}')

        # Update last infered
        self.last_infered = datetime.now(pytz.utc)

        # Fetch data and process - if not converged, fetch more data - break if successfull
        periods = ['2y', '3y', '5y', 'max']
        for period in periods:
            self._fetch(period=period)
            converged = self._data_process()
            if converged:
                break

        # If never converged
        if not converged:
            logging.warning(f'Frac diff for {self.tick} did not converge with period={period} of data')
            return False

        # Predict
        self._predict()

        # Save config file - has to be after predict
        self._save_config()

        return True


    def _assert_tick(self):
        ''' Assert that the tick is available and save meta data '''

        # company_info = self.ticker.info
        company_info = {'EMPTY': None}
        
        if len(company_info) > 1:
            self.compnay_name = company_info['shortName']
            self.exchange = company_info['exchange']
            self.exchange_timezone = company_info['exchangeTimezoneName']
            self.sector = company_info['sector']
            self.currency = company_info['currency']
        else:
            self.compnay_name = self.tick
            self.exchange = 'STO'
            self.exchange_timezone = 'Europe/Stockholm'
            self.sector = 'Unknown'
            self.currency = 'SEK'

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
            # If market is open today but not yet
            if datetime.now(pytz.utc) < schedule_today.market_open:
                schedule_next_day = schedule.iloc[0]
            else: # If market is open today but has closed
                schedule_next_day = schedule.iloc[1]
        else:
            schedule_next_day = schedule.iloc[0]

        next_market_open = schedule_next_day.market_open

        return market_open, next_market_open


    def _fetch(self, period: str):
        ''' FETCH HIST DATA '''

        fetched = False
        while not fetched:
            try:
                self.df = self.ticker.history(period=period, interval='1d')
                fetched = True
            except:
                logging.warning(f'Not able to fetch {self.tick}, will retry in 2 seconds')
                time.sleep(2)
        
        # Save original
        self.df_org = self.df.copy()


    def get_last(self):
        return self.df_org.iloc[-1]


    def get_daily_return(self):
        try:
            df_today = self.df_org.loc[str(datetime.now(pytz.utc).date())]
            _daily_return = (df_today.Open.iloc[-1] / df_today.Close.iloc[0]) - 1
            return _daily_return
        except:
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

        # Feature engineer
        self.df = self.feature_engineer.first_process(self.df)
        self.df, converged = self.feature_engineer.second_process(self.df)
        if not converged:
            return False

        # Truncate to num_steps
        self.df = self.df.iloc[len(self.df)-self.num_steps::]

        # Assert for length
        assert len(self.df) == self.num_steps, f'Dataframe for {self.tick} has length {len(self.df)}, requires {self.num_steps}'

        # Reset and save index
        self.date_index = self.df.index.copy()
        self.df = self.df.reset_index(drop=False)

        # Choose columns for model input
        self.df = self.feature_engineer.select_columns(self.df)

        # Round values - min/max scaler doesnt max/min at 1/0 (1.00000001)
        self.df = self.df.round(decimals=5)

        # Assert for scaling
        for col in self.df.columns:
            assert self.df[col].max() <= 1.0 , f'In {self.tick}, maximum value in column {col} is {self.df[col].max()}, requires <= 1'
            assert self.df[col].min() >= 0.0 , f'In {self.tick}, minimum value in column {col} is {self.df[col].min()}, requires >= 0'

        return True


    def _predict(self):

        ''' Prediction algo '''
        current_price = round(self.df_org.Close.iloc[-1], 3)

        # The last date in the df has to be today
        if self.df_org.index[-1].date() != datetime.now().date():
            logging.error(f'Last date in df is not the same as today for {self.tick}')
            assert False, 'Last date in df is not the same as today'

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
        self.confidence = max(prediction)
        action = np.argmax(prediction)
        self.trigger.set(f'Net: {action}', action, override=False)

        # Model certainty threshold
        if self.trigger.action == 1:
            if max(prediction) < 0.80:
                self.trigger.set(f'Net thrs ({str(round(max(prediction),2))}): {action} -> 0', 0)

        ''' MACD assertion '''
        # if self.trigger.action==1 and self.df.MACDh_12_26_9.iloc[-1]>0:
        #     self.trigger.set(f'MACDh ({str(round(self.df.MACDh_12_26_9.iloc[-1],2))}): buy -> hold', 0)
        # elif self.trigger.action==2 and self.df.MACDh_12_26_9.iloc[-1]<0:
        #     self.trigger.set(f'MACDh ({str(round(self.df.MACDh_12_26_9.iloc[-1],2))}): sell -> hold', 0)

        ''' Stop loss assertion - only if a buy action has been performed '''
        # if self.trigger.action != 2 and self.gain.position=='in' and self.gain.gain>=0.005:
        #     self.trigger.set(f'InvStopLoss ({str(round(self.gain.gain,3))}): {self.trigger.action} -> 2', 2)
        # elif self.trigger.action != 2 and self.gain.position=='in' and self.gain.gain<-0.02:
        #     self.trigger.set(f'StopLoss ({str(round(self.gain.gain,3))}): {self.trigger.action} -> 2', 2)

        ''' Compare trigger action to current position '''
        if self.trigger.action == 1 and self.gain.position == 'in':
            self.trigger.set(f'Trigger {self.trigger.action} redundant -> hold', 0)
        elif self.trigger.action == 2 and self.gain.position == 'out':
            self.trigger.set(f'Trigger {self.trigger.action} redundant -> hold', 0)

        ''' --- COMMITED TO ACTION FROM THIS POINT --- '''
        if self.trigger.action == 1:
            self.gain.buy(current_price)
            self.trigger.set_id()
        elif self.trigger.action == 2:
            self.gain.sell()


class Trigger:

    def __init__(self):
        self.id = 0.0
        self.desc = None
        self.override = False
        self.action = None
        self.model_confidence = 0
        self.set_action_desc()
        
    def set(self, description, action, override=False):
        if not self.override:
            self.desc = description
            self.override = override
            self.action = action
            self.set_action_desc()

    def set_id(self):
        # Each buy/sell trigger pair has an unique ID
        self.id = time.time()

    def reset(self):
        self.desc = None
        self.override = False
        self.model_confidence = 0
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

    net_name = '1630867357_lstmRegul1e-3_DSmax_lstmUnits1024_convFilters1024_lre-7'
    net_path = Path.cwd() / 'networks' / f'{net_name}.tflite'

    net = tflite.Interpreter(model_path=str(net_path))
    net.allocate_tensors()

    tick = TickUnit(tick='ERIC-B.ST', net=net)
    
    
    tick.infer()
    print(tick.trigger.action,tick.trigger.action_desc)
    print(f'---> EOL: {__file__}')