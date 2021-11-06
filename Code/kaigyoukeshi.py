filename="data/extrated_tweet.tsv"
goalname="data/extrated_tweet2.tsv"

import re

file = open(filename)
lines=file.readlines()
print(lines)
file2 = open(goalname,'w')
for i,line in enumerate(lines):
    if re.search(r'\d+\t+.*?\t+',line) == None:
        if re.search(r'\D+?\t+.*?\n+',line) == None:
            print(line)
            lines[i]=re.sub(r'\n',' ',line)
print(lines)
for line in lines:
    
    file2.write(line)
