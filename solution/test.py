import pandas as pd
import numpy as np

user_activity = pd.read_csv('../input/user_activity_log.txt',header=None,sep='\t',names=['user_id','activity_day','page','video_id','author_id','action_type'])

temp_activity = user_activity[(user_activity['activity_day'] >= 1) & (user_activity['activity_day'] <= 16)]

def test(row):
    res=0
    for x in row[:-1]:
        if(x%7==0|(x+1)%7==0):
            res+=1
    print(res)
    print(row[:-1])

weekend_count=temp_activity.groupby(['user_id'])['activity_day'].agg(test)#在周末的总共活动次数
print(0)