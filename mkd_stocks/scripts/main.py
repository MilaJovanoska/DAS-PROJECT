import requests
from bs4 import BeautifulSoup
import sqlite3
import pandas as pd
from datetime import datetime
from multiprocessing import Pool, Lock, Manager
import time

db_lock = Lock()  # Lock for database synchronization when accessed by multiple processes

def get_issuers():
    url = 'https://www.mse.mk/mk/stats/symbolhistory/KMB'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    issuers = [
        option['value'] for option in soup.select('select#Code option')
        if not any(char.isdigit() for char in option['value'])
    ]
    return issuers

def check_last_date(issuer):
    with sqlite3.connect('stock_data.db') as conn:
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS stock_prices (
            issuer TEXT,
            date TEXT,
            last_price TEXT,
            max TEXT,
            min TEXT,
            average TEXT,
            percent_change TEXT,
            quantity TEXT,
            best_trade TEXT,
            total_trade TEXT,
            PRIMARY KEY (issuer, date)
        );''')
        cursor.execute('''SELECT MAX(date(substr(date, 7, 4) || '-' || substr(date, 4, 2) || '-' || substr(date, 1, 2)))
                FROM stock_prices
                WHERE issuer = ?;
                ''', (issuer,))
        result = cursor.fetchone()
        return result[0] if result[0] else '2013.01.01'

def fetch_missing_data(issuer, last_date):
    start_year = int(last_date[:4])
    end_year = datetime.now().year
    all_data = []

    with requests.Session() as session:
        for year in reversed(range(start_year, end_year + 1)):
            url = f'https://www.mse.mk/mk/stats/symbolhistory/{issuer}/?FromDate=01.01.{year}&ToDate=31.12.{year}'
            try:
                response = session.get(url)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, "html.parser")
                table = soup.find('tbody')

                if not table:
                    break

                rows = table.find_all('tr')
                data = [
                    [col.text.strip() for col in row.find_all('td')]
                    for row in rows if len(row.find_all('td')) == 9
                ]
                if data:
                    df = pd.DataFrame(data, columns=['Date', 'LastPrice', 'Max', 'Min', 'Average',
                                                     'PercentChange', 'Quantity', 'BestTrade', 'TotalTrade'])
                    all_data.append(df)
            except requests.exceptions.RequestException as e:
                print(f"Error fetching data for {issuer} for year {year}: {e}")
                continue

    if all_data:
        return pd.concat(all_data)
    return pd.DataFrame(columns=['Date', 'LastPrice', 'Max', 'Min', 'Average',
                                 'PercentChange', 'Quantity', 'BestTrade', 'TotalTrade'])

def save_data_to_db(data, issuer):
    with db_lock:  # Lock to avoid conflicts in the database
        with sqlite3.connect('stock_data.db') as conn:
            cursor = conn.cursor()
            rows_to_insert = [
                (issuer, row['Date'], row['LastPrice'], row['Max'], row['Min'], row['Average'],
                 row['PercentChange'], row['Quantity'], row['BestTrade'], row['TotalTrade'])
                for _, row in data.iterrows()
            ]
            cursor.executemany('''INSERT OR IGNORE INTO stock_prices (issuer, date, last_price, max, min, average, percent_change, quantity, best_trade, total_trade)
                                  VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', rows_to_insert)
            conn.commit()

def process_issuer(issuer):
    last_date = check_last_date(issuer)
    data = fetch_missing_data(issuer, last_date)
    if not data.empty:
        save_data_to_db(data, issuer)

def pipeline():
    issuers = get_issuers()
    with Pool(processes=20) as pool:
        pool.map(process_issuer, issuers)

def main():
    start_time = time.time()
    pipeline()
    end_time = time.time()
    print(f"Data processing completed in {end_time - start_time} seconds")

if __name__ == "__main__":
    main()
