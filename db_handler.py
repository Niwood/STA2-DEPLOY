import pandas as pd
import pandas.io.sql as sqlio
import numpy as np
import json
from datetime import datetime

from utils.utils import secure_fetch

import psycopg2
import yfinance as yf




class Database:

    def __init__(self):

        # DB credentials
        db_credentials = json.load(open('db_credentials.json'))


        # Establish db connection and set cursor
        self.conn = psycopg2.connect(
            host=db_credentials['DB_HOST'],
            dbname=db_credentials['DB_NAME'],
            user=db_credentials['DB_USER'],
            password=db_credentials['DB_PASSW']
            )
        self.cursor = self.conn.cursor()


        # Check daily tick prices in db
        # self.start_date = '2021-01-01'
        self.start_date = datetime(2021,1,1)


    def fill_ticks(self, ticks:list):

        # Query all daily entries
        query = '''
                select tick, date
                from public.daily_tickers
                '''

        df_daily = sqlio.read_sql_query(query, self.conn)

        for tick in ticks:
            df_daily_tick = df_daily[df_daily.tick == tick]

            # 
            if df_daily_tick.empty:
                ticker = yf.Ticker(tick)
                df_fetched = self._fetch(ticker)
                print(df_fetched)
                quit()
                
                query_values = list()
                for index, row in df_fetched[['Close', 'Open', 'Volume']].iterrows():

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

                # print(query_values), quit()
                # self.execute(query)
                # self.disconnect()
                # quit()



    @secure_fetch
    def _fetch(self, ticker):
        return ticker.history(start=self.start_date, end=datetime.now().date(), interval='1d')#.tz_convert('UTC')


    def read(self):
        query = ''' 
                select *
                from public.trigger_buy
                LIMIT 1000
                '''

        a = sqlio.read_sql_query(query, self.conn)

        print(type(a))
        print(a)


    def add_column(self):
        table_name = 'public.daily_tickers'
        column_name = 'date'
        datatype = 'date'
        query = f'''
                ALTER TABLE {table_name}
                ADD {column_name} {datatype};
                '''

        # query = f'''
        # ALTER TABLE {table_name}
        # DROP COLUMN {column_name};
        # '''
                

        self.execute(query)


    def insert_trigger(self, action:int, tick:str, price:float, currency:str, date:datetime, time:datetime, _return:float, model_version:int):
        
        if action == 1:
            table_name = 'public.trigger_buy'
        elif action == 2:
            table_name = 'public.trigger_sell'

        query = f'''
                INSERT INTO {table_name}(
                    tick,
                    price,
                    currency,
                    date,
                    time,
                    return,
                    return_perc,
                    model_version
                    )
                VALUES(
                    {tick},
                    {price},
                    {currency},
                    {date},
                    {time},
                    {_return}
                    {_return*100}
                    {model_version}
                    );
                '''

        self.execute(query)


    def execute(self, query):
        self.cursor.execute(query)
        self.conn.commit()


    def disconnect(self):
        try:
            self.cursor.close()
            self.conn.close()
            print('Disconnected')
        except:
            pass



if __name__ == '__main__':

    tickers = [
        'ERIC-B.ST',
        'VOLV-B.ST',
        'AZN.ST'
        # 'SAND.ST'
        ]

    sta2db = Database()
    try:
        # sta2db.add_column()
       
        pass
    except Exception as e:
        print(e)
        
    sta2db.fill_ticks(ticks=tickers)
    sta2db.disconnect()