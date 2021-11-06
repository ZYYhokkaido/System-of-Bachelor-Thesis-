#This code creates the dataset from Corpus.csv which is downloadable from the
#internet well known dataset which is labeled manually by hand. But for the text
#of tweets you need to fetch them with their IDs.
import tweepy

# Twitter Developer keys here
# It is CENSORED
consumer_key = 'vzkVupmn3UaZYMDFmus46WboQ'
consumer_key_secret = 'x98yyMJrUxf8uuSjh2dutpEvCwnkFcqiS9plfj5mbXDSclRXuA'
access_token = '3737898144-n0ihrxBS8SEEtfyjkGocmtrGpND7eIWZxXZy3W0'
access_token_secret = 'dTFqbZG9R6ejLyN49Dxgc72BI80G1j60VQJkCib00cTEZ'

auth = tweepy.OAuthHandler(consumer_key, consumer_key_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

# This method creates the training set
def createTrainingSet(corpusFile, targetResultFile):
    import csv
    import time

    corpus = []

    with open(corpusFile, 'r') as csvfile:
        lineReader = csv.reader(csvfile, delimiter=',', quotechar="\"")
        for row in lineReader:
            print(row)
            if row[4]=='1' and row[3]=='0' and row[5]=='0' and row[6]=='0' and row[7]=='0':
                corpus.append({"category": row[1], "tweet_id": row[2], "polarity":1})
            if row[4]=='0' and row[3]=='0' and row[5]=='1' and row[6]=='0' and row[7]=='0':
                corpus.append({"category": row[1], "tweet_id": row[2], "polarity":-1})
            if row[4]=='0' and row[3]=='0' and row[5]=='0' and row[6]=='1' and row[7]=='0':
                corpus.append({"category": row[1], "tweet_id": row[2], "polarity":0})

    sleepTime = 1
    trainingDataSet = []

    count_nega=0
    count_posi=0
    count_neu=0

    for tweet in corpus:
        if tweet['polarity']==1 and count_posi!=5000 or tweet['polarity']==-1 and count_nega!=5000 or tweet['polarity']==0 and count_neu!=5000:
            try:
                tweetFetched = api.get_status(tweet["tweet_id"])
                tweet["text"] = tweetFetched.text
                trainingDataSet.append(tweet)
                if tweet['polarity']==1:
                    count_posi+=1
                    print("count_posi=",count_posi)
                if tweet['polarity']==-1:
                    count_nega+=1    
                    print("count_nega=",count_nega)
                if tweet['polarity']==0:
                    count_neu+=1
                    print("count_neu=",count_neu)
                time.sleep(sleepTime)

            except Exception as e:
                print('Error:',e)
                continue
        if count_nega == 5000 and count_posi == 5000 and count_neu == 5000:
            break


    with open(targetResultFile, 'a') as csvfile:
        linewriter = csv.writer(csvfile, delimiter='\t', quotechar="\"")
        for tweet in trainingDataSet:
            try:
                linewriter.writerow([tweet["category"], tweet["text"], tweet["polarity"]])
            except Exception as e:
                print(e)
    return trainingDataSet

# Code starts here
# This is corpus dataset
corpusFile = "data/tweets_open.csv"
# This is my target file
targetResultFile = "data/extrated_tweet.csv"
# Call the method
resultFile = createTrainingSet(corpusFile, targetResultFile)