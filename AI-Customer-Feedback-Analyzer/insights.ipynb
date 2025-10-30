# Insights Notebook
# - Load cleaned_feedback.csv
# - Compute simple recurring issue detection by keyword frequency
# - Simulate sentiment score trend and forecast with ARIMA

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

df = pd.read_csv('cleaned_feedback.csv')
# simple keyword frequency
keywords = ['delivery','packag','support','crash','damag','missing','interface']
kw_counts = {}
for k in keywords:
    kw_counts[k] = df['cleaned_feedback'].str.contains(k).sum()
print("Keyword counts:", kw_counts)

# sentiment score (heuristic)
def score(x):
    txt = str(x)
    if any(w in txt for w in ['excellent','amazing','great','satisfied','helpful','fast']): return 1
    if any(w in txt for w in ['terrible','bad','poor','frustrat','missing','damag','crash']): return -1
    return 0

df['sentiment_score'] = df['cleaned_feedback'].apply(score)
# aggregate by month-like periods (simulated)
series = df['sentiment_score'].rolling(window=3).mean().fillna(0)
model = ARIMA(series, order=(1,1,0))
res = model.fit()
fcast = res.forecast(steps=3)
print("Forecast:", fcast)
plt.plot(series, label='score')
plt.plot(range(len(series), len(series)+3), fcast, '--', label='forecast')
plt.legend()
plt.savefig('sentiment_trend.png')
print("Saved sentiment_trend.png")
