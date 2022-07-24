import json
import os

import pandas as pd
import glob

def summarize_results():
    for method in ['GA', 'EP']:
        d_path = f'results/{method}'
        for path in glob.glob(f'{d_path}/2022**/', recursive=True):
            # params
            with open(f'{path}param.json', 'r') as f:
                results = json.load(f)

            # results
            df = pd.read_csv(f'{path}fitness.csv')
            fitness = df.tail(1).values
            results['repeat'] = fitness[0][0]
            results['fitness'] = fitness[0][1]

            # 書き込み
            filename = f'{d_path}/results.csv'
            if not os.path.isfile(filename):
                with open(filename, 'w') as f:
                    for k in results.keys():
                        f.write(f'{k},')
                    f.write(f'\n')
            else:
                with open(filename, 'a') as f:
                    for k, v in results.items():
                        f.write(f'{v},')
                    f.write(f'\n')


if __name__ == '__main__':
    summarize_results()