import pandas as pd
import numpy as np
cre = pd.read_csv('d://cretest.csv')
act = pd.read_csv("d://acttest.csv")
def get_cre_feature(row):
    user = list(row["user_id"])[0]
    feature = pd.Series()
    feature["user_id"]=user
    if user in watch_sum_count:
        feature['wacth_sum_count']=watch_sum_count[user]
        feature['watch_num_people'] = watch_num_people[user]

        # feature['watch_gbuser_count_mean'] = watch_user_count_mean[user]
        # feature['watch_gbuser_count_var'] = watch_user_count_var[user]
        # feature['watch_gbuser_count_max'] = watch_user_count_max[user]
        # feature['watch_gbuser_count_min'] = watch_user_count_min[user]
        # feature['watch_gbuser_count_skew'] = watch_user_count_skew[user]
        # feature['watch_gbuser_count_kurt'] = watch_user_count_kurt[user]
        #
        # feature['watch_gbuser_day_mean'] = watch_user_day_mean[user]
        # feature['watch_gbuser_day_var'] = watch_user_day_var[user]
        # feature['watch_gbuser_day_max'] = watch_user_day_max[user]
        # feature['watch_gbuser_day_min'] = watch_user_day_min[user]
        # feature['watch_gbuser_day_skew'] = watch_user_day_skew[user]
        # feature['watch_gbuser_day_kurt'] = watch_user_day_kurt[user]
        #
        # feature['watch_gbvideo_count_mean'] = watch_video_count_mean[user]
        # feature['watch_gbvideo_count_var'] = watch_video_count_var[user]
        # feature['watch_gbvideo_count_max'] = watch_video_count_max[user]
        # feature['watch_gbvideo_count_min'] = watch_video_count_min[user]
        # feature['watch_gbvideo_count_skew'] = watch_video_count_skew[user]
        # feature['watch_gbvideo_count_kurt'] = watch_video_count_kurt[user]
        #
        # feature['watch_gbvideo_day_mean'] = watch_video_day_mean[user]
        # feature['watch_gbvideo_day_var'] = watch_video_day_var[user]
        # feature['watch_gbvideo_day_max'] = watch_video_day_max[user]
        # feature['watch_gbvideo_day_min'] = watch_video_day_min[user]
        # feature['watch_gbvideo_day_skew'] = watch_video_day_skew[user]
        # feature['watch_gbvideo_day_kurt'] = watch_video_day_kurt[user]
        # print(watch_num_people[user])
    else:
        feature['wacth_sum_count'] = 0
        feature['watch_num_people'] = 0

        # feature['watch_gbuser_count_mean'] = 0
        # feature['watch_gbuser_count_var'] = 0
        # feature['watch_gbuser_count_max'] =0
        # feature['watch_gbuser_count_min'] = 0
        # feature['watch_gbuser_count_skew'] = 0
        # feature['watch_gbuser_count_kurt'] = 0
        #
        # feature['watch_gbuser_day_mean'] = 0
        # feature['watch_gbuser_day_var'] = 0
        # feature['watch_gbuser_day_max'] = 0
        # feature['watch_gbuser_day_min'] = 0
        # feature['watch_gbuser_day_skew'] = 0
        # feature['watch_gbuser_day_kurt'] = 0
        #
        # feature['watch_gbvideo_count_mean'] = 0
        # feature['watch_gbvideo_count_var'] =0
        # feature['watch_gbvideo_count_max'] = 0
        # feature['watch_gbvideo_count_min'] = 0
        # feature['watch_gbvideo_count_skew'] = 0
        # feature['watch_gbvideo_count_kurt'] = 0
        #
        # feature['watch_gbvideo_day_mean'] = 0
        # feature['watch_gbvideo_day_var'] = 0
        # feature['watch_gbvideo_day_max'] = 0
        # feature['watch_gbvideo_day_min'] = 0
        # feature['watch_gbvideo_day_skew'] = 0
        # feature['watch_gbvideo_day_kurt'] = 0
    # print(list(row['user_id'])[0])
    # print(get_author_count(list(row['user_id'])[0]))
    return feature
creater=np.unique(cre['user_id'])
watch_sum_count={}
watch_num_people={}
# watch_user_count_mean={}
# watch_user_count_var={}
# watch_user_count_max={}
# watch_user_count_min={}
# watch_user_count_skew={}
# watch_user_count_kurt={}
#
# watch_user_day_mean={}
# watch_user_day_var={}
# watch_user_day_max={}
# watch_user_day_min={}
# watch_user_day_skew={}
# watch_user_day_kurt={}
#
# watch_video_count_mean={}
# watch_video_count_var={}
# watch_video_count_max={}
# watch_video_count_min={}
# watch_video_count_skew={}
# watch_video_count_kurt={}
#
# watch_video_day_mean={}
# watch_video_day_var={}
# watch_video_day_max={}
# watch_video_day_min={}
# watch_video_day_skew={}
# watch_video_day_kurt={}
def get_watch_user_feature(row):
    feature=pd.Series()
    feature['user_id']=list(row["user_id"])[0]
    feature['user_count']=len(row["user_id"])
    feature['user_day']=len(np.unique(row["activity_day"]))
    # feature['0_user_count'] = len(row[row.page==0])
    # feature['1_user_count'] = len(row[row.page == 1])
    # feature['2_user_count'] = len(row[row.page == 2])
    # feature['3_user_count'] = len(row[row.page == 3])
    # feature['4_user_count'] = len(row[row.page == 4])
    # feature['0_user_day'] = len(np.unique(row[row.page == 0]["activity_day"]))
    # feature['1_user_day'] = len(np.unique(row[row.page == 1]["activity_day"]))
    # feature['2_user_day'] = len(np.unique(row[row.page == 2]["activity_day"]))
    # feature['3_user_day'] = len(np.unique(row[row.page == 3]["activity_day"]))
    # feature['4_user_day'] = len(np.unique(row[row.page == 4]["activity_day"]))
    return feature

def get_watch_video_feature(row):
    feature=pd.Series()
    feature['user_id']=list(row["user_id"])[0]
    feature['video_count']=len(row["user_id"])
    feature['video_day']=len(np.unique(row["activity_day"]))
    # feature['0_video_count'] = len(row[row.page==0])
    # feature['1_video_count'] = len(row[row.page == 1])
    # feature['2_video_count'] = len(row[row.page == 2])
    # feature['3_video_count'] = len(row[row.page == 3])
    # feature['4_video_count'] = len(row[row.page == 4])
    # feature['0_video_day'] = len(np.unique(row[row.page == 0]["activity_day"]))
    # feature['1_video_day'] = len(np.unique(row[row.page == 1]["activity_day"]))
    # feature['2_video_day'] = len(np.unique(row[row.page == 2]["activity_day"]))
    # feature['3_video_day'] = len(np.unique(row[row.page == 3]["activity_day"]))
    # feature['4_video_day'] = len(np.unique(row[row.page == 4]["activity_day"]))
    return feature

def get_author_feature(row):
    global creater
    # print(author[0])
    author=list(row["author_id"])[0]
    if author in creater:
        watch_sum_count[author] = len(row["user_id"])
        watch_num_people[author] = len(np.unique(row["user_id"]))

        # print(row)
        user_feature = row.groupby("user_id", sort=True).apply(get_watch_user_feature)
        video_feature=row.groupby("user_id", sort=True).apply(get_watch_video_feature)
        # watch_user_count_mean[author]= np.mean(user_feature['user_count'])
        # watch_user_count_var[author] = np.var(user_feature['user_count'])
        # watch_user_count_max[author] = np.max(user_feature['user_count'])
        # watch_user_count_min[author]= np.min(user_feature['user_count'])
        # watch_user_count_skew[author]= pd.Series(user_feature['user_count']).skew
        # watch_user_count_kurt[author] = pd.Series(user_feature['user_count']).kurt
        #
        # watch_user_day_mean[author] = np.mean(user_feature['user_day'])
        # watch_user_day_var[author] = np.var(user_feature['user_day'])
        # watch_user_day_max[author] = np.max(user_feature['user_day'])
        # watch_user_day_min[author] = np.min(user_feature['user_day'])
        # watch_user_day_skew[author] = pd.Series(user_feature['user_day']).skew
        # watch_user_day_kurt[author] = pd.Series(user_feature['user_day']).kurt
        #
        # watch_video_count_mean[author] = np.mean(video_feature['video_count'])
        # watch_video_count_var[author] = np.var(video_feature['video_count'])
        # watch_video_count_max[author] = np.max(video_feature['video_count'])
        # watch_video_count_min[author] = np.min(video_feature['video_count'])
        # watch_video_count_skew[author] = pd.Series(video_feature['video_count']).skew
        # watch_video_count_kurt[author] = pd.Series(video_feature['video_count']).kurt
        #
        # watch_video_day_mean[author] = np.mean(video_feature['video_day'])
        # watch_video_day_var[author] = np.var(video_feature['video_day'])
        # watch_video_day_max[author] = np.max(video_feature['video_day'])
        # watch_video_day_min[author] = np.min(video_feature['video_day'])
        # watch_video_day_skew[author] = pd.Series(video_feature['video_day']).skew
        # watch_video_day_kurt[author] = pd.Series(video_feature['video_day']).kurt
        # print(user_feature)

act.groupby("author_id",sort=True).apply(get_author_feature)
feature=cre.groupby("user_id",sort=True).apply(get_cre_feature)
print(feature)