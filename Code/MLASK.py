import csv
from mlask import MLAsk

originalFile='/Users/zhangyiyang/Desktop/毕业论文相关/日本語データ/Lesson_analysis.tsv'

emotion_analyzer = MLAsk()

corpus=[]
counter=0
sum1=0
with open(originalFile,'r') as o:
    linereader=csv.reader(o,delimiter='\t',quotechar=' ')
    for tweet in linereader:
        if tweet[2]!='0':
            if emotion_analyzer.analyze(tweet[1])['emotion']!=None:
                if emotion_analyzer.analyze(tweet[1])['orientation']=='POSITIVE':
                    sum1+=1
                    if tweet[2]=='1':
                        counter+=1
                    
                    
                    
            
print(sum1,counter)


