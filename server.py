import traceback
import sys

import numpy as np
import pandas as pd
from tickunit import TickUnit
import time
import schedule
import json
from pathlib import Path
from datetime import datetime, timedelta
import logging
logging.basicConfig(
    filename='logs/server_logger.log',
    filemode='w',
    level=logging.INFO,
    format='%(levelname)s: %(asctime)s - %(message)s'
    )

from utils.external_features import ExternalFeatures

import psycopg2
import pytz
import smtplib, ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from colorama import init as colorama_init
colorama_init()
from colorama import Fore, Back, Style
from tabulate import tabulate

import tflite_runtime.interpreter as tflite


class Array:
    '''
    Production array
    Input:
        tickers = list of tickers
        update_rate = in minutes


    '''
    logging.info('Server started')

    def __init__(self, tickers):

        self.DEBUG = True

        # Parameters
        self.net_name = '1630867357_lstmRegul1e-3_DSmax_lstmUnits1024_convFilters1024_lre-7'
        self.respect_market_hours = True
        self.mute = False
        self.tickers = tickers
        self.report = {
            'tick':None,
            'last_infered':None,
            'price':None,
            'action':None,
            'trigger_description': None,
            'gain': 0,
            'position': None,
            'market_open': False,
            'next_market_open': False
            }

        # Paths
        self.status_files_path = Path.cwd() / 'status_files'
        self.net_path = Path.cwd() / 'networks' / f'{self.net_name}.tflite'

        # External features
        self.extern_features = ExternalFeatures()

        # Read DB credentials
        db_credentials = json.load(open('db_credentials.json'))

        # Establish db connection and set cursor
        self.conn = psycopg2.connect(
            host=db_credentials['DB_HOST'],
            dbname=db_credentials['DB_NAME'],
            user=db_credentials['DB_USER'],
            password=db_credentials['DB_PASSW']
            )
        self.cursor = self.conn.cursor()

        # Load flite model
        self.net = self.load_network()

        # Start ticker units
        self._start_ticker_units()

        # Metadata
        self.tz_local = pytz.timezone('Europe/Stockholm')


    def _start_ticker_units(self):
        ''' Init ticker units list - run on init '''
        self.units = list()
        for tick in self.tickers:
            self.units.append(TickUnit(tick=tick, net = self.net))
            logging.info(f'Ticker unit initiated for {tick}')
        logging.info('All ticker units started')


    def load_network(self):
        ''' Load network and allocate tensors '''
        net = tflite.Interpreter(model_path=str(self.net_path))
        net.allocate_tensors()
        return net


    def _convert_action(self, action):
        if action==1:
            return 'BUY'
        elif action==2:
            return 'SELL'
        else:
            return 'HOLD'


    def _notify(self, subject, text=None, append_report=True):
        ''' Notifications handling '''

        if self.mute: #No notification if mute is True
            return

        if time.time() - self.t0 < 60*5: #No notification if the uptime is less
            print(f'No notification sent: Server uptime is less than 60*5 sec ({time.time() - self.t0} sec)')
            return

        sender_email = "HampusBackStockNotification@gmail.com"  # Enter your address
        password = 'eGiNqS0qc58U'
        receiver_email = ["hampus.back@gmail.com", 'robinniwood@gmail.com']  # Enter receiver address
        # receiver_email = ['robinniwood@gmail.com']
        

        # Message
        message = MIMEMultipart("alternative")
        message["Subject"] = 'STA2-PROD server notification: ' + subject
        message["From"] = sender_email
        message["To"] = ", ".join(receiver_email)

        
        body = """\
        This message is sent from STA2-PROD server.\n

    
        """
        body += text + '''
        \n
        
        
        '''
        if append_report:
            for k,v in self.report.items():
                body+= f'   {k}: {v}\n'
            

        message.attach(MIMEText(body, "plain"))
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
            server.login(sender_email, password)
            server.sendmail(
                sender_email, receiver_email, message.as_string()
            )

        logging.info(f'Notification has been sent: {datetime.now(self.tz_local)} {self.tz_local.zone}')


    def run(self):
        logging.info('Server is running')
        self.t0 = time.time()

        # schedule.every(1).hours.do(self.extern_features.update)
        # schedule.every(30).minutes.do(self.infer_units)
        # schedule.run_all()

        # MAIN LOOP
        while True:
        
            # RUN SCHEDUELE
            # schedule.run_pending()
            self.extern_features.update()
            self.infer_units()

            # IDLE CONDITION - If no market is open, sleep until earliest opening
            if not any(self.market_open_all) and self.respect_market_hours:
                next_market_open_min = min(self.next_market_open_all)
                logging.info(f'Market is closed and inference is set to idle. Next market opening: {next_market_open_min} UTC | {next_market_open_min.tz_convert(self.tz_local)} {self.tz_local.zone}')

                while True:
                    seconds_until_start = (next_market_open_min - datetime.now(pytz.utc)).seconds
                    if seconds_until_start <= 0:
                        logging.info(f'The market is now open')
                        break
                    
                    time.sleep(1)
                    
            # Sleep 1s for performance
            time.sleep(60)


    def infer_units(self):
        t0_infer_all = time.time()
        

        # Save market openings and next opening
        self.market_open_all = list()
        self.next_market_open_all = list()

        for unit in self.units:

            # Save market open/close status
            market_open, next_market_open = unit.check_market_open()
            self.market_open_all.append(market_open)
            self.next_market_open_all.append(next_market_open)

            # Infer if the market is open
            if market_open and self.respect_market_hours:
                try:
                    t0_infer = time.time()
                    converged = unit.infer()
                    if not converged:
                        continue
                    logging.info(f'Successfull inference on {unit.tick}: {unit.trigger.action}. Execution time {int(time.time()-t0_infer)} seconds')
                except Exception as e:
                    logging.error(f'Error during inference {unit.tick}. {e} {traceback.print_tb(e.__traceback__)}')
            else:
                continue
                
            # Insert into db
            if unit.trigger.action == 1:
                self._buy_query(unit)
                logging.info(f'Buy triggered for {unit.tick}')
            elif unit.trigger.action == 2:
                self._sell_query(unit)
                logging.info(f'Sell triggered for {unit.tick}')
                
        logging.info(f'Inference completed. Total time: {int(time.time()-t0_infer_all)} seconds, {int((time.time()-t0_infer_all)/len(self.units))} seconds/tick')


    def _buy_query(self, unit):

        query = f'''
                INSERT INTO public.trigger (
                    trigger_id,
                    tick,
                    buy_price,
                    buy_date,
                    buy_time,
                    trigger_desc_buy,
                    model_confidence_buy,
                    debug
                )
                VALUES (
                    {unit.trigger.id},
                    '{unit.tick}',
                    {unit.gain.buy_price},
                    '{str(datetime.now().date())}',
                    '{str(datetime.now().time()).split('.')[0]}',
                    '{unit.trigger.desc}',
                    {unit.confidence},
                    {self.DEBUG}
                    )
                '''
        self._execute(query)


    def _sell_query(self, unit):

        query = f'''
            UPDATE public.trigger
            SET
                sell_price = {unit.get_last().Close},
                sell_date = '{str(datetime.now().date())}',
                sell_time = '{str(datetime.now().time()).split('.')[0]}',
                return = {unit.get_last().Close - unit.gain.buy_price},
                return_perc = {unit.gain.gains[-1]},
                trigger_desc_sell = '{unit.trigger.desc}',
                model_confidence_sell = {unit.confidence},
                debug = {self.DEBUG}
            WHERE trigger_id = {unit.trigger.id}
            '''
        self._execute(query)


    def _execute(self, query):
        ''' Execute the query to the DB
        trigger_id numeric PRIMARY KEY,
        tick varchar,
        buy_price numeric,
        sell_price numeric,
        buy_date date,
        buy_time time,
        sell_date date,
        sell_time time,
        return numeric,
        return_perc numeric,
        trigger_desc_buy varchar,
        trigger_desc_sell varchar,
        model_confidence_buy numeric,
        model_confidence_sell numeric,
        debug BOOLEAN NOT NULL
        '''
        self.cursor.execute(query)
        self.conn.commit()


    def disconnect(self):
        ''' Disconnect from DB '''
        self.cursor.close()
        self.conn.close()
        logging.warning('Disconnected - terminating server')



if __name__ == '__main__':

    # tickers = [
    #     'ERIC-B.ST',
    #     'VOLV-B.ST',
    #     'AZN.ST']

    tickers = json.load(open('tick_symbols.json'))['ticks']

    array = Array(tickers=list(set(tickers)))
    array.mute = True #True if notifications should be muted
    array.respect_market_hours = True #True if inference should be idle off-market
    try:
        array.run()
    except KeyboardInterrupt:
        array.disconnect()
    print(f'---> EOL: {__file__}')