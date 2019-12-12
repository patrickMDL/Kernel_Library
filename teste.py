import pandas as pd


df = pd.read_csv('/home/patrick/DownloadsFMEL_Dataset.csv', dtype={
   'id': int,
   'season': str,
   'division': int,
   'round': int,
   'localTeam': str,
   'visitorTeam': str,
   'localGoals': int,
   'visitorGoals': int,
   'fecha': str,
   'date': str,
})

df['season'] = df['season'].str.split('-').str[0].astype(int)
print(df.dtypes)