import pandas as pd
from sklearn.datasets import load_iris

iris = load_iris()
train = pd.DataFrame(iris.data, columns=iris.feature_names)
train['label'] = iris.target
train = train.sample(frac=1, random_state=1)
train_df = train.iloc[:120]
test_df = train.iloc[120:]
train_df.to_csv('train_tmp.csv', index=False)
test_df.to_csv('test_tmp.csv', index=False)
print('csv saved')

import subprocess
subprocess.run(['python', 'discriminant_analysis.py', '--train', 'train_tmp.csv', '--test', 'test_tmp.csv', '--output-report', 'report.txt', '--output-test', 'test_pred.csv'], check=True)

print('report\n', open('report.txt', 'r', encoding='utf-8').read())
print('test_pred head\n', pd.read_csv('test_pred.csv').head())
