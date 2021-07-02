import traceback
import sys

import numpy as np
import pandas as pd
from tickunit import TickUnit
import time
import json
from pathlib import Path
from datetime import datetime
import pytz
import smtplib, ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from colorama import init as colorama_init
colorama_init()
from colorama import Fore, Back, Style
from tabulate import tabulate



class Array:
    '''
    Production array
    Input:
        tickers = list of tickers
        update_rate = in minutes


    '''

    def __init__(self, tickers, update_rate=10):

        # Parameters
        self.respect_market_hours = True
        self.mute = False
        self.tickers = tickers
        self._start_ticker_units()
        self.update_rate = update_rate
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

        # Metadata
        self.tz_local = pytz.timezone('Europe/Stockholm')

        # Paths
        self.status_files_path = Path.cwd() / 'status_files'
        


    def _start_ticker_units(self):
        ''' Init ticker units list - run on init '''
        self.units = list()
        for tick in self.tickers:
            self.units.append(TickUnit(tick=tick))


    def _notify(self, subject, text=None, append_report=True):
        ''' Notifications handling '''

        if self.mute: return f'Mute = {self.mute}'
        
        sender_email = "HampusBackStockNotification@gmail.com"  # Enter your address
        password = 'eGiNqS0qc58U'
        receiver_email = ["hampus.back@gmail.com", 'robinniwood@gmail.com']  # Enter receiver address
        # receiver_email = 'robinniwood@gmail.com'
        

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


    def run(self):
        # self._notify('Unit test', text='Test' append_report=True), quit()
        while True:
            # Save market openings and next opening
            market_open_all = list()
            next_market_open_all = list()

            # Reset table
            table_body = list()
            table_headers = [
                'TICK',
                'EXC',
                'POS',
                'TRIGGER',
                'TRIGGER DESC',
                'PRICE',
                'POSITION GAIN',
                'D.RETURN',
                'MARKET'
                ]
            
            try:
                for unit in self.units:

                    # Check market is opening
                    market_open, next_market_open = unit.check_market_open()
                    market_open_all.append(market_open)
                    next_market_open_all.append(next_market_open)

                    # Infer if the market is open
                    if market_open and self.respect_market_hours:
                        unit.infer()
                        
                    # Get trigger
                    trigger = unit.trigger
                    if trigger.action in (0, None):
                        _trigger_desc = Back.YELLOW + Fore.BLACK + trigger.action_desc + Style.RESET_ALL
                    elif trigger.action in (1,2):
                        _trigger_desc = Back.RED + Fore.BLACK + trigger.action_desc + Style.RESET_ALL

                    # Reporting
                    self.report['tick'] = unit.tick
                    self.report['last_infered'] = str(unit.last_infered)
                    self.report['price'] = round(unit.get_last().Close,1)
                    self.report['action'] = trigger.action
                    self.report['trigger_description'] = trigger.desc
                    self.report['gain'] = unit.gain.gain
                    self.report['position'] = unit.gain.position
                    self.report['market_open'] = market_open
                    self.report['next_market_open'] = next_market_open


                    '''
                    'TICK',
                    'EXCHANGE',
                    'POSITION',
                    'ACTION',
                    'TRIGGER DESC'
                    'LAST PRICE',
                    'POSITION GAIN',
                    'DAILY RETURN',
                    'MARKET'
                    '''
                    _daily_return = round(unit.get_daily_return()*100,1)
                    table_body.append([
                        unit.tick,
                        unit.exchange,
                        Fore.YELLOW + unit.gain.position + Style.RESET_ALL if unit.gain.position=='out' else Back.YELLOW + Fore.BLACK + unit.gain.position + Style.RESET_ALL,
                        _trigger_desc,
                        trigger.desc,
                        f'{unit.currency} {str(round(unit.get_last().Close,1))}',
                        Back.GREEN + Fore.BLACK + str(round(unit.gain.gain*100,2)) + '%' + Style.RESET_ALL if unit.gain.gain>=0 else Back.RED + Fore.BLACK + str(round(unit.gain.gain*100,2)) + '%' + Style.RESET_ALL,
                        Fore.GREEN + str(_daily_return) + '%' + Style.RESET_ALL if _daily_return>=0 else Fore.RED + str(_daily_return) + '%' + Style.RESET_ALL,
                        Back.GREEN + Fore.BLACK + 'Open' + Style.RESET_ALL if market_open else Back.RED + Fore.BLACK + 'Closed' + Style.RESET_ALL
                        ])
                    
                    # Notification
                    hours_since_last_notification = (unit.last_notification_sent - datetime.now(pytz.utc)).seconds / 1800
                    if trigger.action in (1,2) and hours_since_last_notification >= 0.5 :
                        self._notify('Trigger', text='', append_report=True)
                        unit.last_notification_sent = datetime.now(pytz.utc)

                # Save table data to json
                with open(self.status_files_path / 'table.json', 'w') as outfile:
                    json.dump(
                        {'table_body':table_body, 'table_headers':table_headers, 'table_id':time.time()},
                        outfile
                        )
                # Save inference info to json
                with open(self.status_files_path / 'inference_info.json', 'w') as outfile:
                    json.dump(
                        {'idle':False, 'last_inference':str(datetime.now(pytz.utc))},
                        outfile
                        )
  

            except Exception as e:
                print(traceback.print_tb(e.__traceback__))
                self._notify(
                    'Error during inference',
                    text=f'''
                    Error during inference {datetime.now(pytz.utc)} UTC: {repr(e)} \n


                    Full traceback: \n 
                    {traceback.print_tb(e.__traceback__)}
                    ''',
                    append_report=False)



            # If no market is open, sleep until earliest opening
            if not any(market_open_all) and self.respect_market_hours:
                
                next_market_open_min = min(next_market_open_all)
                print(f'Market is closed and inference is set to idle.')
                print(f'Next market opening: {next_market_open_min} UTC | {next_market_open_min.tz_convert(self.tz_local)} {self.tz_local.zone}')

                while True:

                    seconds_until_start = (next_market_open_min - datetime.now(pytz.utc)).seconds - (self.update_rate*60)
                    if seconds_until_start <= 0:
                        break

                    day = seconds_until_start // (24 * 3600)
                    seconds_until_start = seconds_until_start % (24 * 3600)
                    hour = seconds_until_start // 3600
                    seconds_until_start %= 3600
                    minutes = seconds_until_start // 60
                    seconds_until_start %= 60
                    seconds = seconds_until_start

                    # Save inference info to json
                    with open(self.status_files_path / 'inference_info.json', 'w') as outfile:
                        json.dump(
                            {'idle':True, 'next_market_open':str(next_market_open_min), 'day':day, 'hour':hour, 'minutes':minutes, 'seconds':seconds},
                            outfile
                            )


                    time.sleep(1)
                    



            # Sleep according to update rate
            time.sleep(self.update_rate*60)





if __name__ == '__main__':
    tickers = ['ERIC-B.ST', 'VOLV-B.ST', 'AZN.ST', 'SAND.ST', 'TEL2-B.ST', 'HM-B.ST', 'SEB-A.ST', 'INVE-A.ST', 'LUNE.ST']
    # tickers = ['SAND.ST']
    update_rate = 1

    array = Array(tickers=tickers, update_rate=update_rate)
    array.mute = False #True if notifications should be muted
    array.respect_market_hours = True #True if inference should be idle off-market
    array.run()
    print(f'---> EOL: {__file__}')