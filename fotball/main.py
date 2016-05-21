import pandas as pd



df = pd.read_csv('data/E1.csv')


properties = ["FTHG","FTAG","FTR","HTHG","HTAG","HTR","HS","AS","HST","AST","HC","AC","HF","AF","AY","HR","AR"]
df =  df[properties]

print df.iloc[0]