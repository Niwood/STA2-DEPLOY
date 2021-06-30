import traceback

import numpy as np
import pandas as pd
from tickunit import TickUnit
import time
from datetime import datetime
import pytz
import pprint
import smtplib, ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from colorama import init as colorama_init
colorama_init()
from colorama import Fore, Back, Style

class Array:
    '''
    Production array
    Input:
        tickers = list of tickers
        update_rate = in minutes


    '''

    def __init__(self, tickers, update_rate=10):

        self.mute = False
        self.tickers = tickers
        self.update_rate = update_rate
        self.report = {
            'tick':None,
            'last_infered':None,
            'price':None,
            'action':None,
            'trigger_description': None,
            'gain': 0,
            'position': None
            }

        self._start_ticker_units()
        


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
            
            try:
                for unit in self.units:
                    trigger = unit.infer()
                    self.report['tick'] = unit.tick
                    self.report['last_infered'] = str(unit.last_infered)
                    self.report['price'] = round(unit.get_last().Close,1)
                    self.report['action'] = trigger.action
                    self.report['trigger_description'] = trigger.desc
                    self.report['gain'] = unit.gain.gain
                    self.report['position'] = unit.gain.position

                    if trigger.action in (1,2):
                        print(Fore.RED)
                    pprint.pprint(self.report)
                    print(Style.RESET_ALL)
                    print('-'*10)

                    hours_since_last_notification = (unit.last_notification_sent - datetime.now(pytz.utc)).seconds / 1800
                    if trigger.action in (1,2) and hours_since_last_notification >= 0.5 :
                        self._notify('Trigger', text='', append_report=True)
                        unit.last_notification_sent = datetime.now(pytz.utc)

            except Exception as e:
                print(Fore.RED + traceback.print_tb(e.__traceback__))
                print(Style.RESET_ALL)
                self._notify(
                    'Error during inference',
                    text=f'''
                    Error during inference {datetime.now(pytz.utc)} UTC: {repr(e)} \n


                    Full traceback: \n 
                    {traceback.print_tb(e.__traceback__)}
                    ''',
                    append_report=False)

            print(Fore.BLUE + f'Inference completed at {datetime.now(pytz.utc)} UTC')
            print(Style.RESET_ALL)

            # Sleep according to update rate
            time.sleep(self.update_rate*60)





if __name__ == '__main__':
    tickers = ['ERIC-B.ST', 'VOLV-B.ST', 'AZN.ST', 'SAND.ST', 'TEL2-B.ST', 'HM-B.ST', 'SEB-A.ST', 'INVE-A.ST', 'AZN.ST', 'LUNE.ST']
    # tickers = ['VOLV-B.ST']
    update_rate = 1

    array = Array(tickers=tickers, update_rate=update_rate)
    array.mute = False #True if notifications should be muted
    array.run()
    print(f'---> EOL: {__file__}')