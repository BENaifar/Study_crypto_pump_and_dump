"""data.binance.vision archive client. [orig: Data/Binance/utility.py]

Only change vs. the original (R3a): the fallback store directory (originally ``dirname(utility.py)``)
is injected via ``store_directory``. ``STORE_DIRECTORY`` (env) and ``folder`` keep their original priority.
"""

from __future__ import annotations

import json
import os
import re
import sys
import urllib.error
import urllib.request
from argparse import ArgumentTypeError
from datetime import date
from pathlib import Path

from pnd.data.binance.constants import BASE_URL, DAILY_INTERVALS, END_DATE, START_DATE


def download_monthly_klines(trading_type, symbols, num_symbols, intervals, years, months, start_date, end_date, folder,
                            checksum, store_directory=None):
    """Q16. [orig: utility.py:L9-L45]"""
    current = 0
    date_range = None

    if start_date and end_date:
        date_range = start_date + " " + end_date
    if not start_date:
        start_date = START_DATE
    else:
        start_date = convert_to_date_object(start_date)
    if not end_date:
        end_date = END_DATE
    else:
        end_date = convert_to_date_object(end_date)

    for symbol in symbols:

        for interval in intervals:
            for year in years:
                for month in months:
                    current_date = convert_to_date_object('{}-{}-01'.format(year, month))

                    if current_date >= start_date and current_date <= end_date:
                        path = get_path(trading_type, "klines", "monthly", symbol, interval)
                        file_name = "{}-{}-{}-{}.zip".format(symbol.upper(), interval, year, '{:02d}'.format(month))
                        download_file(path, file_name, date_range, folder, store_directory)

                    # QUIRK(Q16): checksum download sits outside the date filter [orig: L39-L43]
                    if checksum == 1:
                        checksum_path = get_path(trading_type, "klines", "monthly", symbol, interval)
                        checksum_file_name = "{}-{}-{}-{}.zip.CHECKSUM".format(symbol.upper(), interval, year,
                                                                               '{:02d}'.format(month))
                        download_file(checksum_path, checksum_file_name, date_range, folder, store_directory)

        current += 1


def download_daily_klines(trading_type, symbols, num_symbols, intervals, dates, start_date, end_date, folder, checksum,
                          store_directory=None):
    """[orig: utility.py:L48-L84] (ported, unused by the pipeline)"""
    current = 0
    date_range = None

    if start_date and end_date:
        date_range = start_date + " " + end_date
    if not start_date:
        start_date = START_DATE
    else:
        start_date = convert_to_date_object(start_date)
    if not end_date:
        end_date = END_DATE
    else:
        end_date = convert_to_date_object(end_date)

    # Get valid intervals for daily
    intervals = list(set(intervals) & set(DAILY_INTERVALS))
    print("Found {} symbols".format(num_symbols))

    for symbol in symbols:
        print("[{}/{}] - start download daily {} klines ".format(current + 1, num_symbols, symbol))

        for interval in intervals:
            for date_ in dates:
                current_date = convert_to_date_object(date_)

                if current_date >= start_date and current_date <= end_date:
                    path = get_path(trading_type, "klines", "daily", symbol, interval)
                    file_name = "{}-{}-{}.zip".format(symbol.upper(), interval, date_)
                    download_file(path, file_name, date_range, folder, store_directory)

                if checksum == 1:
                    checksum_path = get_path(trading_type, "klines", "daily", symbol, interval)
                    checksum_file_name = "{}-{}-{}.zip.CHECKSUM".format(symbol.upper(), interval, date_)
                    download_file(checksum_path, checksum_file_name, date_range, folder, store_directory)

        current += 1


def get_destination_dir(file_url, folder=None, store_directory=None):
    """[orig: utility.py:L86-L92]; ``store_directory`` replaces ``dirname(realpath(__file__))``."""
    env_store_directory = os.environ.get('STORE_DIRECTORY')
    if folder:
        env_store_directory = folder
    if not env_store_directory:
        env_store_directory = str(store_directory)
    return os.path.join(env_store_directory, file_url)


def get_download_url(file_url):
    return "{}{}".format(BASE_URL, file_url)


def get_all_symbols(type):
    """[orig: utility.py:L99-L106]"""
    if type == 'um':
        response = urllib.request.urlopen("https://fapi.binance.com/fapi/v1/exchangeInfo").read()
    elif type == 'cm':
        response = urllib.request.urlopen("https://dapi.binance.com/dapi/v1/exchangeInfo").read()
    else:
        response = urllib.request.urlopen("https://api.binance.com/api/v3/exchangeInfo").read()
    return list(map(lambda symbol: symbol['symbol'], json.loads(response)['symbols']))


def download_file(base_path, file_name, date_range=None, folder=None, store_directory=None):
    """Q15. [orig: utility.py:L109-L149]"""
    download_path = "{}{}".format(base_path, file_name)
    if folder:
        base_path = os.path.join(folder, base_path)
    if date_range:
        date_range = date_range.replace(" ", "_")
        base_path = os.path.join(base_path, date_range)
    save_path = get_destination_dir(os.path.join(base_path, file_name), folder, store_directory)

    if os.path.exists(save_path):
        return

    # make the directory
    # QUIRK(Q15): existence is checked relative to the process cwd, creation under the store dir [orig: L123-L124]
    if not os.path.exists(base_path):
        Path(get_destination_dir(base_path, None, store_directory)).mkdir(parents=True, exist_ok=True)

    try:
        download_url = get_download_url(download_path)
        dl_file = urllib.request.urlopen(download_url)
        length = dl_file.getheader('content-length')
        # QUIRK(Q15): without content-length `blocksize` is undefined -> NameError escapes [orig: L130-L132]
        if length:
            length = int(length)
            blocksize = max(4096, length // 100)

        with open(save_path, 'wb') as out_file:
            dl_progress = 0
            while True:
                buf = dl_file.read(blocksize)
                if not buf:
                    break
                dl_progress += len(buf)
                out_file.write(buf)
                done = int(50 * dl_progress / length)
                sys.stdout.write("\r[%s%s]" % ('#' * done, '.' * (50 - done)))
                sys.stdout.flush()

    except urllib.error.HTTPError:
        pass


def convert_to_date_object(d):
    year, month, day = [int(x) for x in d.split('-')]
    date_obj = date(year, month, day)
    return date_obj


def get_start_end_date_objects(date_range):
    start, end = date_range.split()
    start_date = convert_to_date_object(start)
    end_date = convert_to_date_object(end)
    return start_date, end_date


def match_date_regex(arg_value, pat=re.compile(r'\d{4}-\d{2}-\d{2}')):
    if not pat.match(arg_value):
        raise ArgumentTypeError
    return arg_value


def get_path(trading_type, market_data_type, time_period, symbol, interval=None):
    """[orig: utility.py:L186-L194]"""
    trading_type_path = 'data/spot'
    if trading_type != 'spot':
        trading_type_path = f'data/futures/{trading_type}'
    if interval is not None:
        path = f'{trading_type_path}/{time_period}/{market_data_type}/{symbol.upper()}/{interval}/'
    else:
        path = f'{trading_type_path}/{time_period}/{market_data_type}/{symbol.upper()}/'
    return path
