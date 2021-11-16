import pandas as pd
import pandas.io.sql as sqlio
import numpy as np
import json
from datetime import datetime, timedelta
import time
import logging
logging.basicConfig(
    filename='logs/live_feed_logger.log',
    filemode='w',
    level=logging.INFO,
    format='%(levelname)s: %(asctime)s - %(message)s'
    )

# from utils.utils import secure_fetch

import psycopg2
import yfinance as yf




class LiveFeed:
    logging.info('Live feed started')
    '''
    Running script to update daily prices of stocks
    '''

    def __init__(self):

        # Debug mode
        self.DEBUG = False

        # Parameters
        self.UPDATE_FREQUENCY = 1*3600 # Server update frequency in seconds
        self.HEARTBEAT_FREQUENCY = 60*30 # Heartbeat frequency in seconds
        self.UPDATE_RECOIL = 30 # Number of days in the past to update
        self.BASELINE_START_DATE = datetime(2021,1,1) # Start date for all new ticks that will be fetched

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

        # Run server
        try:
            self.run()
        except Exception as e:
            logging.warning('Exception occurred in run(): ', exc_info=True)
            self.disconnect()


    def run(self):
        '''
        Run the server
        '''

        counter = 0
        while True:

            # Heartbeat
            if counter % self.HEARTBEAT_FREQUENCY == 0 or self.DEBUG:
                logging.info('HEARTBEAT')

            # Read tick symbols and fill for empty ticks
            if counter % 10 == 0 or self.DEBUG:
                self.read_tick_symbols()
                self.fill_for_empty_ticks()

            # Update the ticks
            if counter % self.UPDATE_FREQUENCY == 0 or self.DEBUG:
                self.update_daily()
                logging.info('Completed update')
                
            # Sleep and step counter
            time.sleep(1)
            counter += 1


    def read_tick_symbols(self):
        '''
        Read json for tick symbols
        '''

        done = False
        while not done:
            try:
                self.ticks = json.load(open('tick_symbols.json'))['ticks']
                done = True
            except:
                logging.warning('Was not able to read tick symbols from tick_symbols.json')
                time.sleep(5)


    def update_daily(self):
        '''
        Updated the values for each tick to the DB
        '''

        for tick in self.ticks:

            # Fetch data
            start_date = datetime.now() - timedelta(days=self.UPDATE_RECOIL)
            df_fetched = self._fetch(tick, start_date)
            df_fetched = df_fetched[['Open', 'Close', 'Volume']]
            df_fetched.sort_index(ascending=False, inplace=True)
            df_fetched.rename(
                columns={'Open': 'open_price', 'Close': 'close_price', 'Volume': 'volume'},
                inplace=True)

            # Read all days from DB
            query = f'''
                    select *
                    from public.daily_tickers
                    WHERE tick = '{tick}'
                    '''
            try:
                df_read = sqlio.read_sql_query(query, self.conn)
            except Exception as e:
                logging.warning(f'Was not able to fetch data for {tick}, will continue with next tick: {e}')
                continue

            df_read.index = pd.to_datetime(df_read.date)
            df_read.drop(['date', 'id', 'tick'], axis=1, inplace=True)
            df_read.sort_index(ascending=False, inplace=True)

            # Truncate the fetched values from the last dated value in DB
            last_day_in_db = df_read.index.max()
            df_to_add = df_fetched.loc[df_fetched.index > last_day_in_db]
            df_to_update = df_fetched.loc[(df_fetched.index <= last_day_in_db) & (df_fetched.index >= last_day_in_db - timedelta(days=1))]

            # Update entries - format to sql and execute for each row
            for index, row in df_to_update.iterrows():
                query = self._format_sql_update(tick, index, row)
                self._execute(query)
                logging.info(f'Updated data for {tick} Date={str(index.date())} Close={round(row.close_price,1)}')

            # Add missing entries - format to sql query for insert
            if not df_to_add.empty:
                query = self._format_sql_insert(df_to_add, tick)
                try:
                    self._execute(query)
                    logging.info(f'Added missing data for {tick}')
                except Exception as e:
                    logging.warning(f'Was not able to update {tick}: {e}')


    def fill_for_empty_ticks(self):
        '''
        Fill all daily entries (from baseline_start_date)
        if there is no entries in the DB for a particular tick
        '''

        for tick in self.ticks:

            # Query and read a tick
            query = f'''
                    SELECT tick, date
                    FROM public.daily_tickers
                    WHERE tick = '{tick}'
                    '''
            df_tick = sqlio.read_sql_query(query, self.conn)

            # Check if the df is empty
            if df_tick.empty:
                logging.info(f'Recognized new tick symbol {tick}')

                # If empty - fetch data and setup the query
                df_fetched = self._fetch(tick, self.BASELINE_START_DATE)
                df_fetched.drop(['High', 'Low', 'Dividends', 'Stock Splits'], axis=1, inplace=True)
                df_fetched.rename(
                    columns={'Open': 'open_price', 'Close': 'close_price', 'Volume': 'volume'},
                    inplace=True)

                # Format to sql query for insert
                query = self._format_sql_insert(df_fetched, tick)

                # Execute
                self._execute(query)
                logging.info(f'Fetched successfully {tick}')


    def _fetch(self, tick, start):
        '''
        Fetch tick data from yfinance
        '''
        
        ticker = yf.Ticker(tick)

        done = False
        while not done:
            try:
                return ticker.history(start=start, interval='1d')#.tz_convert('UTC')
            except:
                logging.warning(f'Not able to fetch data for {tick}')
                time.sleep(5)


    def _format_sql_insert(self, df, tick):
        '''
        Format a dataframe to sql query
        '''

        query_values = list()
        for index, row in df[['close_price', 'open_price', 'volume']].iterrows():

            # Convert to list
            value_item = row.tolist()

            # Insert tick name and date values to the list
            value_item.insert(0, tick)
            value_item.insert(len(value_item), str(index.date()))

            # Append as tuple to query_values
            query_values.append(str(tuple(value_item)))
        
        # Convert query_values to str and replace quotation marks
        query_values = str(query_values)
        query_values = query_values.replace('"','')
        query_values = query_values[1:-1]

        # Final query for insertion
        query = f'''
                INSERT INTO public.daily_tickers (
                    tick,
                    close_price,
                    open_price,
                    volume,
                    date
                )
                VALUES
                {query_values}

        '''
        return query


    def _format_sql_update(self, tick, index, row):
        ''' 
        Format SQL query for updating existing entries - returns list for each entry 
        '''

        query = f'''
            UPDATE public.daily_tickers
            SET
                close_price = {row.close_price},
                open_price = {row.open_price},
                volume = {row.volume}
            WHERE tick = '{tick}' AND date = '{str(index.date())}'
            '''
        
        return query


    def _execute(self, query):
        ''' Execute the query to the DB '''

        if self.DEBUG:
            print('DEBUG ON: Executed to DB')
        else:
            self.cursor.execute(query)
            self.conn.commit()


    def disconnect(self):
        ''' Disconnect from DB '''
        self.cursor.close()
        self.conn.close()
        logging.warning('Disconnected - terminating server')
        quit()



if __name__ == '__main__':


    live_feed = LiveFeed()
    live_feed.disconnect()