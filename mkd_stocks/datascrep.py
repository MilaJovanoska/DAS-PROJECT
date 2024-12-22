import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
from multiprocessing import Pool, Lock, Manager
import time
import os
import django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'DAS_PROJECT.settings')
django.setup()
from mkd_stocks.models import Stock

def get_issuers():
    url = 'https://www.mse.mk/mk/stats/symbolhistory/KMB'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    issuers = [
        option['value'] for option in soup.select('select#Code option')
        if not any(char.isdigit() for char in option['value'])
    ]
    return issuers

db_lock = Lock()
def fetch_missing_data(issuer):
    start_year = int("2014")
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
                    for row in data:
                        new_stock = Stock(
                            issuer=issuer,
                            date=row[0],
                            last_price=row[1],
                            max=row[2],
                            min=row[3],
                            average=row[4],
                            percent_change=row[5],
                            quantity=row[6],
                            best_trade=row[7],
                            total_trade=row[8]
                        )
                        with db_lock:
                             if not Stock.objects.filter(issuer=issuer, date=row[0]).exists():
                                new_stock.save()


            except requests.exceptions.RequestException as e:
                print(f"Error fetching data for {issuer} for year {year}: {e}")
                continue

    if all_data:
        return pd.concat(all_data)
    return pd.DataFrame(columns=['Date', 'LastPrice', 'Max', 'Min', 'Average',
                                 'PercentChange', 'Quantity', 'BestTrade', 'TotalTrade'])


def process_issuer(issuer):
    fetch_missing_data(issuer)


def pipeline():
    issuers = get_issuers()
    with Pool(processes=9) as pool:
        pool.map(process_issuer, issuers)


def main():
    start_time = time.time()
    pipeline()
    end_time = time.time()
    print(f"Data processing completed in {end_time - start_time} seconds")


if __name__ == "__main__":
    main()
    print("HELOOO")
