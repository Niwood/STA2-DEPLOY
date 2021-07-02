from colorama import init as colorama_init
colorama_init()
from colorama import Fore, Back, Style
from tabulate import tabulate

import sys
import os
from pathlib import Path
import json
from datetime import datetime, timezone
import pytz
import time


class Status:
    ''' 
    Reads server status from table_files
    and prints to console
    '''
    def __init__(self):
        # Path
        self.status_files_path = Path.cwd() / 'status_files'
        
        self.tz_local = pytz.timezone('Europe/Stockholm')
        self.printed_table_id = None

        self.run()


    def read(self):
        ''' Read json files '''

        # Read table
        with open(self.status_files_path / 'table.json') as jfile:
            self.table_data = json.load(jfile)

        # Read inference info
        with open(self.status_files_path / 'inference_info.json') as jfile:
            self.inference_data = json.load(jfile)


    def run(self):

        while True:
            # Read
            self.read()
            # print(self.inference_data), quit()

            # Print table
            if self.table_data['table_id'] != self.printed_table_id:
                print(tabulate(self.table_data['table_body'], self.table_data['table_headers'], tablefmt="fancy_grid"))
                self.printed_table_id = self.table_data['table_id']
            # print(Fore.MAGENTA + f'Inference completed at {datetime.now(pytz.utc)} UTC' + Style.RESET_ALL)
            # print('\n')


            # # Inference idle
            if self.inference_data['idle']:
                next_market_open = datetime.fromisoformat(self.inference_data['next_market_open'])

                print(Fore.YELLOW)
                print('Market is closed and inference is set to idle.')
                print(f'Next market opening: {next_market_open} UTC | {next_market_open.replace(tzinfo=timezone.utc).astimezone(tz=self.tz_local)} {self.tz_local.zone}')
                while self.inference_data['idle']:
                    
                    day = self.inference_data['day']
                    hour = self.inference_data['hour']
                    minutes = self.inference_data['minutes']
                    seconds = self.inference_data['seconds']

                    print(f'Time until opening: {day} days - {hour} hours - {minutes} min - {seconds} sec', end='\r')
                    
                    time.sleep(2)
                    self.read()
                print(Style.RESET_ALL)

            time.sleep(5)




if __name__ == '__main__':
    Status()