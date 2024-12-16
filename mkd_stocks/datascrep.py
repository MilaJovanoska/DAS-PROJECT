import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
from multiprocessing import Pool, Lock, Manager
import time

from mkd_stocks.models import Stock

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
    print("")

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
                    # TODO
                    # new_stock = Stock("SITE ROWS STO GI IMATE, rows[0]..rows[9]")
                    #
                    # # Save the object to the database
                    df = pd.DataFrame(data, columns=['Date', 'LastPrice', 'Max', 'Min', 'Average',
                                                     'PercentChange', 'Quantity', 'BestTrade', 'TotalTrade'])
                    all_data.append(df)

            # new_stock.save()
            except requests.exceptions.RequestException as e:
                print(f"Error fetching data for {issuer} for year {year}: {e}")
                continue

    if all_data:
        return pd.concat(all_data)
    return pd.DataFrame(columns=['Date', 'LastPrice', 'Max', 'Min', 'Average',
                                 'PercentChange', 'Quantity', 'BestTrade', 'TotalTrade'])

def save_data_to_db(data, issuer):
   Stock.save

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
    print("HELOOO")