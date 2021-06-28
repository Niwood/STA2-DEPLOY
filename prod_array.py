import numpy as np
import pandas as pd
from tickunit import TickUnit
import time
from datetime import datetime
import pprint
import smtplib, ssl

class Array:
    '''
    Production array
    Input:
        tickers = list of tickers
        update_rate = in minutes


    '''

    def __init__(self, tickers, update_rate=10, verbose=False):
        self.tickers = tickers
        self.update_rate = update_rate
        self.report = {
            'tick':None,
            'last_infered':None,
            'price':None,
            'action':None,
            'trigger_description': None
            }

        self._start_ticker_units()
        self.run(verbose=verbose)


    def _start_ticker_units(self):
        ''' Init ticker units list - run on init '''
        self.units = list()
        for tick in self.tickers:
            self.units.append(TickUnit(tick=tick))


    def _notify(self):
        ''' Notifications handling '''
        port = 465  # For SSL
        smtp_server = "smtp.gmail.com"
        # sender_email = "HampusBackStockNotification@gmail.com"  # Enter your address
        sender_email = "robinniwood@gmail.com"  # Enter your address
        receiver_email = ["hampus.back@gmail.com", 'robinniwood@gmail.com', "HampusBackStockNotification@gmail.com"]  # Enter receiver address
        # password = 'ijF5!kkL23'
        password = '60TG5VxnXn42'
        message = """\
        Subject: Test

        This message is sent from Python.\n

        """
        for k,v in self.report.items():
            message+= f'{k}: {v}\n'

        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
            server.login(sender_email, password)
            server.sendmail(sender_email, receiver_email, message)


    def run(self, verbose):

        while True:
            
            for unit in self.units:
                trigger = unit.infer()
                self.report['tick'] = unit.tick
                self.report['last_infered'] = str(unit.last_infered)
                self.report['price'] = round(unit.get_last().Close,1)
                self.report['action'] = trigger.action
                self.report['trigger_description'] = trigger.desc

                if verbose:
                    pprint.pprint(self.report)
                    print('-'*10)

                if trigger.action in (1,2) and not unit.last_notification_sent == datetime.today().date():
                    self._notify()
                    unit.last_notification_sent = datetime.today().date()
            
            print(f'Inference completed at {datetime.today()}')

            # Sleep according to update rate
            time.sleep(self.update_rate*60)





if __name__ == '__main__':
    tickers = ['ERIC-B.ST', 'VOLV-B.ST', 'AZN.ST', 'SAND.ST', 'TEL2-B.ST', 'HM-B.ST', 'SEB-A.ST', 'INVE-A.ST', 'AZN.ST', 'LUNE.ST']
    # tickers = ['VOLV-B.ST']
    update_rate = 1


    Array(tickers=tickers, update_rate=update_rate, verbose=True)
    print(f'---> EOL: {__file__}')