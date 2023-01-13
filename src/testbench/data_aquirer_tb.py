import sys
import os.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from data_aquirer import Data_Aquirer

PATH = 'pairs'
PAIR = 'EURUSD'
MINUTES = 15
API_KEY = 'kvtkOoyqcuTgNrBqRGIhhLe766CLYbpo'
TIME_FORMAT = '%Y-%m-%d'
DATE_START = '2022-11-01'
DATE_END = '2023-01-09'


def main():
    aquirer = Data_Aquirer(PATH, API_KEY, TIME_FORMAT)
    data = aquirer.get(PAIR, MINUTES, DATE_START, DATE_END, save=True)

if __name__ == '__main__':
    main()