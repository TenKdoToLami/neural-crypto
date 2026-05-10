import pandas as pd
import glob
import os

files = glob.glob('data/raw_2022/*.csv')
rois = []

print("Buy & Hold ROI (2022-01-01 to 2023-12-31):")
for f in files:
    df = pd.read_csv(f)
    start_p = df['open'].iloc[0]
    end_p = df['close'].iloc[-1]
    roi = (end_p - start_p) / start_p * 100
    rois.append(roi)
    asset_name = os.path.basename(f).split('_')[0]
    print(f"  {asset_name}: {roi:+.2f}%")

avg_roi = sum(rois) / len(rois)
print(f"\nAverage Buy & Hold Portfolio Return: {avg_roi:+.2f}%")
