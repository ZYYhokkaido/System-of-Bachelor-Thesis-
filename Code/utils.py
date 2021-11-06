import string
import pandas as pd
import unicodedata

def load_dataset(filename,n=5000,state=6):
    df=pd.read_csv(filename,sep='\t')
    print(df)
    # Converts multi-class to binary-class.

    mapping = {5:1,1:-1}
    df=df[(df.star_rating != 2) & (df.star_rating != 3) & (df.star_rating != 4)]
    df.star_rating = df.star_rating.map(mapping)

    #sampling
    # df=df.sample(frac=1,random_state=state)
    # grouped=df.groupby('star_rating')
    
    grouped=df.groupby('star_rating')
    df=grouped.head(n=n)

    return df.review_body.values,df.star_rating.values
