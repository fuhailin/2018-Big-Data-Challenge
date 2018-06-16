#coding=utf-8

import sys
from collections import Counter
from scipy.stats import stats
import pandas as pd
import numpy as np
import math
now_data = '/home/dl/deeplearning/projectofkzl/a3d6_chusai_a_train'
all_dataSet_path = now_data+'/dealed/all_dataSet.csv'
one_dataSet_train_path = now_data+'/dealed_data/one_dataSet_train_'
one_dataSet_test_path = now_data+'/dealed_data/one_dataSet_test_'
two_dataSet_train_path = now_data+'/dealed_data/two_dataSet_train_'
two_dataSet_test_path = now_data+'/dealed_data/two_dataSet_test_'
three_dataSet_train_path = now_data+'/dealed_data/three_dataSet_train_'

train_path = now_data+'/train_and_test/train.csv'
test_path = now_data+'/train_and_test/test.csv'
one_path = now_data+'/train_and_test/one.csv'
two_path = now_data+'/train_and_test/two.csv'

register = 'register.csv'
create = 'create.csv'
launch = 'launch.csv'
activity = 'activity.csv'
startday=1
endday=16


def get_train_label(train_path,test_path):
    train_reg = pd.read_csv(train_path+register,usecols=['user_id'])
    train_cre = pd.read_csv(train_path+create,usecols=['user_id'])
    train_lau = pd.read_csv(train_path+launch,usecols=['user_id'])
    train_act = pd.read_csv(train_path+activity,usecols=['user_id'])
    train_data_id = np.unique(pd.concat([train_reg,train_cre,train_lau,train_act]))

    test_reg = pd.read_csv(test_path+register,usecols=['user_id'])
    test_cre = pd.read_csv(test_path+create,usecols=['user_id'])
    test_lau = pd.read_csv(test_path+launch,usecols=['user_id'])
    test_act = pd.read_csv(test_path+activity,usecols=['user_id'])
    test_data_id = np.unique(pd.concat([test_reg,test_cre,test_lau,test_act]))

    train_label = []
    for i in train_data_id:
        if i in test_data_id:
            train_label.append(1)
        else:
            train_label.append(0)
    train_data = pd.DataFrame()
    train_data['user_id'] = train_data_id
    train_data['label'] =  train_label
    return train_data


def get_test(test_path):
    test_reg = pd.read_csv(test_path+register,usecols=['user_id'])
    rest_cre = pd.read_csv(test_path+create,usecols=['user_id'])
    test_lau = pd.read_csv(test_path+launch,usecols=['user_id'])
    test_act = pd.read_csv(test_path+activity,usecols=['user_id'])
    test_data_id = np.unique(pd.concat([test_reg,rest_cre,test_lau,test_act]))
    test_data = pd.DataFrame()
    test_data['user_id'] = test_data_id
    return test_data


#计算连续天数，数目和最大值
'''def continuous(data):
    maxcount=0
    tempcount=1
    avgcount=0
    count=0
    for i in data:
        if i==1:
            tempcount=tempcount+1
        elif tempcount>1:
            count=count+1
            if tempcount>maxcount:
                maxcount=tempcount
                avgcount=avgcount+tempcount
            tempcount=1
    if tempcount > 1:
        count = count + 1
        if tempcount > maxcount:
            maxcount = tempcount
            avgcount = avgcount + tempcount
    if count==0:
        avgcount=0
    else:
        avgcount=avgcount/(count+0.0)
    return count,maxcount,avgcount'''
def continuous(data):
    tempcount=1
    list_con_daycount=[]
    for i in data:
        if i==1:
            tempcount=tempcount+1
        elif tempcount>1:
            list_con_daycount.append(tempcount)
            tempcount=1
    if tempcount > 1:
        list_con_daycount.append(tempcount)
    return list_con_daycount



def get_create_feature(row):
    feature = pd.Series()
    listcreate=list(row["create_day"])
    #创建视频数目
    viedocount=len(listcreate)
    createday=[]
    createcount=[]
    count=0
    for i in range(viedocount):
        if i==0:
            temp=listcreate[i]
            count=1
        else:
            if listcreate[i]==temp:
                count=count+1
            else:
                createday.append(temp)
                createcount.append(count)
                temp=listcreate[i]
                count=1
    createday.append(temp)
    createcount.append(count)

    createday.append(endday)
    createday = pd.Series(createday)
    Screateday = createday.diff()
    Screateday = pd.Series(Screateday).dropna().reset_index(drop=True)
    feature["user_id"] = list(row["user_id"])[0]
    feature["create_day_mean"] = np.mean(list(Screateday)) if len(Screateday) > 0 else 0
    feature["create_day_var"] = np.var(list(Screateday)) if len(Screateday) > 0 else 0
    feature["create_day_max"] = np.max(list(Screateday)) if len(Screateday) > 0 else 0
    feature["create_day_min"] = np.min(list(Screateday)) if len(Screateday) > 0 else 0
    cre_skew,cre_kurt = calc_stat(list(Screateday))
    feature["create_day_skew"] = cre_skew
    feature["create_day_kurt"] = cre_kurt
    feature["create_day_count"] = len(createday)

    '''row["create_day_mean"] = Screateday.mean()
    row["create_day_var"] = Screateday.var()
    row["create_day_max"] = Screateday.max()
    row["create_day_min"] = Screateday.min()
    row["create_day_skew"] = Screateday.skew()
    row["create_day_kurt"] = Screateday.kurt()
    row["create_day_count"] = createday.count()'''
    length = len(Screateday)
    if length > 0:
        feature["create_day_lastinterval"] = Screateday[length - 1]
    else:
        feature["create_day_lastinterval"] = 0
    if length > 1:
        feature["create_day_last2interval"] = Screateday[length - 2]
    else:
        feature["create_day_last2interval"] = 0
    feature["create_day_precent"] = length / (endday - createday.min() + 1.0)

    list_con_daycount = list(continuous(Screateday))
    feature["create_day_concount"] = len(list_con_daycount)
    feature["create_day_conmaxcount"] = np.max(list_con_daycount) if len(list_con_daycount)>0 else 0
    feature["create_day_conmincount"] = np.min(list_con_daycount) if len(list_con_daycount)>0 else 0
    feature["create_day_conavgcount"] = np.mean(list_con_daycount) if len(list_con_daycount)>0 else 0
    feature["create_day_convarcount"] = np.var(list_con_daycount) if len(list_con_daycount)>0 else 0

    createcount=pd.Series(createcount)
    feature["create_ndcount_sum"] = np.sum(list(createcount)) if len(createcount) > 0 else 0
    feature["create_ndcount_mean"] = np.mean(list(createcount)) if len(createcount) > 0 else 0
    feature["create_ndcount_var"] = np.var(list(createcount)) if len(createcount) > 0 else 0
    feature["create_ndcount_max"] = np.max(list(createcount)) if len(createcount) > 0 else 0
    feature["create_ndcount_min"] = np.min(list(createcount)) if len(createcount) > 0 else 0
    createcount_skew , createcount_kurt = calc_stat(list(createcount))
    row["create_ndcount_skew"] = createcount_skew
    row["create_ndcount_kurt"] = createcount_kurt
    Screatecount=createcount.diff()
    Screatecount = pd.Series(Screatecount).dropna().reset_index(drop=True)
    feature["create_count_sum"] = np.sum(list(Screatecount)) if len(Screatecount) > 0 else 0
    feature["create_count_mean"] = np.mean(list(Screatecount)) if len(Screatecount) > 0 else 0
    feature["create_count_var"] = np.var(list(Screatecount)) if len(Screatecount) > 0 else 0
    feature["create_count_max"] = np.max(list(Screatecount)) if len(Screatecount) > 0 else 0
    feature["create_count_min"] = np.min(list(Screatecount)) if len(Screatecount) > 0 else 0
    screate_skew,screate_kurt = calc_stat(list(Screatecount))
    feature["create_count_skew"] = screate_skew
    feature["create_count_kurt"] = screate_kurt
    feature["create_count_count"] = len(createcount)
    length = len(Screatecount)
    if length > 0:
        feature["create_count_lastinterval"] = Screatecount[length - 1]
    else:
        feature["create_count_lastinterval"] = 0
    if length > 1:
        feature["create_count_last2interval"] = Screatecount[length - 2]
    else:
        feature["create_count_last2interval"] = 0
    
    weekend_create = row[(row.create_day%7==0) | ((row.create_day +1)%7==0)]
    day_count=len(weekend_create['create_day'].drop_duplicates())
    count=weekend_create.__len__()
    feature["create_weekendcount"] = count
    if day_count>0:
        feature["create_mean_weekendcount"] = count/day_count
    else:
        feature["create_mean_weekendcount"] = 0

    '''feature["user_id"] = list(row["user_id"])[0]
    feature["create_day_mean"] = list(row["create_day_mean"])[0]
    feature["create_day_var"] = list(row["create_day_var"])[0]
    feature["create_day_max"] = list(row["create_day_max"])[0]
    feature["create_day_min"] = list(row["create_day_min"])[0]
    feature["create_day_skew"] = list(row["create_day_skew"])[0]
    feature["create_day_kurt"] = list(row["create_day_kurt"])[0]
    feature["create_day_count"] = list(row["create_day_count"])[0]
    feature["create_day_lastinterval"] = list(row["create_day_lastinterval"])[0]
    feature["create_day_last2interval"] = list(row["create_day_last2interval"])[0]
    feature["create_day_precent"] = list(row["create_day_precent"])[0]
    feature["create_day_concount"] = list(row["create_day_concount"])[0]
    feature["create_day_conmaxcount"] = list(row["create_day_conmaxcount"])[0]
    feature["create_day_conavgcount"] = list(row["create_day_conavgcount"])[0]

    feature["create_count_sum"] = list(row["create_count_sum"])[0]
    feature["create_count_mean"] = list(row["create_count_mean"])[0]
    feature["create_count_var"] = list(row["create_count_var"])[0]
    feature["create_count_max"] = list(row["create_count_max"])[0]
    feature["create_count_min"] = list(row["create_count_min"])[0]
    feature["create_count_skew"] = list(row["create_count_skew"])[0]
    feature["create_count_kurt"] = list(row["create_count_kurt"])[0]
    feature["create_count_count"] = list(row["create_count_count"])[0]
    feature["create_count_lastinterval"] = list(row["create_count_lastinterval"])[0]
    feature["create_count_last2interval"] = list(row["create_count_last2interval"])[0]'''

    return feature


def get_register_feature(row):
    feature = pd.Series()
    feature["user_id"] = list(row["user_id"])[0]
    feature["register_type"] = list(row["register_type"])[0]
    feature["device type"] = list(row["device type"])[0]
    return feature


def get_launch_feature(row):
    feature = pd.Series()
    feature["user_id"] = list(row["user_id"])[0]
    #新加
    launch_day = list(row['launch_day'])
    feature["launch_ndmean"] = np.mean(launch_day) if len(launch_day) > 0 else 0
    feature["launch_ndvar"] = np.var(launch_day) if len(launch_day) > 0 else 0
    feature["launch_ndmax"] = np.max(launch_day) if len(launch_day) > 0 else 0
    feature["launch_ndmin"] = np.min(launch_day) if len(launch_day) > 0 else 0
    launch_skew , launch_kurt = calc_stat(launch_day) if len(launch_day) > 0 else 0
    feature["launch_ndskew"] = launch_skew
    feature["launch_ndkurt"] = launch_kurt

    Slaunch_day=pd.Series(row["launch_day"])
    registerday=Slaunch_day.min()
    launch_count=len(Slaunch_day)
    Slaunch_day = Slaunch_day.append(pd.Series([endday])).reset_index(drop=True)
    Slaunch_day = Slaunch_day.diff()
    Slaunch_day = pd.Series(Slaunch_day).dropna().reset_index(drop=True)
    feature["launch_mean"]=np.mean(list(Slaunch_day)) if len(Slaunch_day) > 0 else 0
    feature["launch_var"]=np.var(list(Slaunch_day)) if len(Slaunch_day) > 0 else 0
    feature["launch_max"]=np.max(list(Slaunch_day)) if len(Slaunch_day) > 0 else 0
    feature["launch_min"]=np.min(list(Slaunch_day)) if len(Slaunch_day) > 0 else 0
    launch_skew , launch_kurt = calc_stat(list(Slaunch_day))
    feature["launch_skew"] = launch_skew
    feature["launch_kurt"]=launch_kurt
    feature["launch_count"]=launch_count
    length=len(Slaunch_day)
    if length>0:
        feature["launch_lastinterval"]=Slaunch_day[len(Slaunch_day)-1]
    else:
        feature["launch_lastinterval"] = 0
    if length>1:
        feature["launch_last2interval"] = Slaunch_day[len(Slaunch_day)-2]
    else:
        feature["launch_last2interval"] = 0
    feature["launch_precent"]=launch_count/(endday-registerday+1.0)
    feature["launch_last7days"] = row[(row.launch_day>=(int(endday)-6))&(row.launch_day<=int(endday))].__len__()
    feature["launch_last5days"] = row[(row.launch_day >= (int(endday) - 4)) & (row.launch_day <= int(endday))].__len__()
    feature["launch_last3days"] = row[(row.launch_day >= (int(endday) - 2)) & (row.launch_day <= int(endday))].__len__()
    list_con_daycount = list(continuous(Slaunch_day))
    feature["launch_con_count"] = len(list_con_daycount)
    feature["launch_con_maxcount"] = np.max(list_con_daycount) if len(list_con_daycount) > 0 else 0
    feature["launch_con_avgcount"] = np.mean(list_con_daycount) if len(list_con_daycount) > 0 else 0
    feature["launch_con_varcount"] = np.var(list_con_daycount) if len(list_con_daycount) > 0 else 0

    weekend_launch = row[(row.launch_day%7==0) | ((row.launch_day +1)%7==0)]
    day_count=len(weekend_launch['launch_day'].drop_duplicates())
    count=weekend_launch.__len__()
    feature["launch_weekendcount"] = count
    if day_count>0:
        feature["launch_mean_weekendcount"] = row[(row.launch_day%7==0) | ((row.launch_day +1)%7==0)].__len__()/day_count
    else:
        feature["launch_mean_weekendcount"] = 0
    return feature

'''
  feature["register_type"] = list(row["register_type"])[0]
    feature["device type"] = list(row["device type"])[0]
'''



'''
['launch_mean','launch_var','launch_max', 'launch_min','launch_skew', 'launch_kurt', 'launch_count' ,'launch_lastinterval', 'launch_last2interval',
 'launch_precent', 'launch_last7days', 'create_day_skew',
 'launch_last5days', 'launch_last3days' ',launch_con_count',
 'launch_con_maxcount', 'create_day_mean', 'create_day_min',
 'create_day_var', 'create_day_max', 'create_day_kurt',
'create_day_count', 'create_day_lastinterval', 'create_day_last2interval', 'create_day_precent', 'create_day_concount', 'create_day_conmaxcount'
, 'create_count_sum', 'create_count_mean', 'create_count_var', 'create_count_max', 'create_count_min', 'create_count_skew',
 'create_count_kurt', 'create_count_count', 'create_count_lastinterval', 'create_count_last2interval', 'register_type', 'device type'
 , 'activity_day_sum', 'activity_day_max', 'activity_day_min', 'activity_day_mean', 'activity_day_std', 'activity_day_kurt', 'activity_day_skew'
, 'activity_day_last', '0_action_num', '1_action_num', '2_action_num', '3_action_num', '4_action_num']

'''

def calc(data):
    n = len(data)
    if n==0:
        return [0,0,0]
    niu = 0.0
    niu2 = 0.0
    niu3 = 0.0
    for a in data:
        niu += a
        niu2 += a**2
        niu3 += a**3
    niu/= n   #这是求E(X)
    niu2 /= n #这是E(X^2)
    niu3 /= n #这是E(X^3)
    sigma = math.sqrt(niu2 - niu*niu) #这是D（X）的开方，标准差
    return [niu,sigma,niu3] #返回[E（X）,标准差，E（X^3）]


def calc_stat(data):
    [niu,sigma,niu3] = calc(data)
    n = len(data)
    if n==0:
        return [0,0]
    niu4 = 0.0
    for a in data:
        a -= niu
        niu4 += a ** 4
    niu4 /= n
    if sigma==0:
        skew=0
        kurt=0
    else:
        skew = (niu3 - 3*niu*sigma**2 - niu**3)/(sigma**3)
        kurt =  niu4/(sigma**2)
    return [skew,kurt] #返回了均值，标准差，偏度，峰度


def final_act(data):
    for item in reversed(data):
        if item > 0:
            return item
    return 0


def get_activity_feature(row):
    feature = pd.Series()
    feature['user_id'] = list(row['user_id'])[0]
    # print(feature['user_id'])
    activity_day_count=len(np.unique(row['activity_day']))
    feature['activity_day_sum'] = activity_day_count
    feature['activity_day_max'] = row['activity_day'].value_counts().max()
    feature['activity_day_min'] = row['activity_day'].value_counts().min()
    feature['activity_day_mean'] = row['activity_day'].value_counts().mean()
    feature['activity_day_std'] = row['activity_day'].value_counts().std()
    feature['activity_day_kurt'] = row['activity_day'].value_counts().kurt()
    feature['activity_day_skew'] = row['activity_day'].value_counts().skew()
    feature['activity_day_last'] = row['activity_day'].value_counts()[list(row['activity_day'])[-1]]
    for i in range(5):
        if i in row['action_type'].value_counts().index:
            feature['{}_action_num'.format(i)]  = row['action_type'].value_counts()[i]
        else:
            feature['{}_action_num'.format(i)] = 0
    
    feature['0_page_count'] = len(row[(row.user_id==feature['user_id']) & (row.page==0)])
    feature['1_page_count'] = len(row[(row.user_id == feature['user_id']) & (row.page == 1)])
    feature['2_page_count'] = len(row[(row.user_id == feature['user_id']) & (row.page == 2)])
    feature['3_page_count'] = len(row[(row.user_id == feature['user_id']) & (row.page == 3)])
    feature['4_page_count'] = len(row[(row.user_id == feature['user_id']) & (row.page == 4)])
    amount = len(row[row.user_id==feature['user_id']])
    # print("i am very sad")
    # print("happy"+str(amount),int(feature['0_page_count']))
    feature['activity_count_sum']=amount
    feature['0_page_count_div_sum'] = int(feature['0_page_count']) / amount
    # print("what the error?")
    feature['1_page_count_div_sum'] = int(feature['1_page_count']) / amount
    feature['2_page_count_div_sum'] = int(feature['2_page_count']) / amount
    feature['3_page_count_div_sum'] = int(feature['3_page_count']) / amount
    feature['4_page_count_div_sum'] = int(feature['4_page_count']) / amount
    # print("test1")
    video_list = list(row['video_id'])
    video_set = set(video_list)
    max_video_amount = 0
    for item in video_set:
        max_video_amount = max(max_video_amount,item)
    feature['video_id_mode'] = max_video_amount
    author_list = list(row['author_id'])
    author_set = set(author_list)
    max_author_amount = 0
    # print("test2")
    for item in author_set:
        max_author_amount = max(max_author_amount,item)
    feature['author_id_mode'] = max_author_amount

    activity_author_count=len(row['author_id'].drop_duplicates())
    feature['activity_author_count'] = activity_author_count
    if activity_author_count>0:
        feature['activity_author_meancount']=amount/ activity_author_count
    else:
        feature['activity_author_meancount']=0


    # print("test3")
    #'activity_page0_mean', 'activity_page0_std'等等
    page_list = list(row['page'])
    day_list = list(row['activity_day'])
    page_dict = [{},{},{},{},{}]
    for index in range(startday,endday+1):
        for index2 in range(5):
            page_dict[index2][index] = 0
    for index in range(len(page_list)):
        inner_key = day_list[index]
        page_dict[page_list[index]][inner_key] += 1

    # print("test4")
    #page0
    feature['activity_page0_mean'] = np.mean(list(page_dict[0].values()))
    feature['activity_page0_std'] = np.var(list(page_dict[0].values()))
    feature['activity_page0_max'] = np.max(list((page_dict[0].values())))
    feature['activity_page0_min'] = np.min(list(page_dict[0].values()))
    [skew,kurt] = calc_stat(page_dict[0].values())
    feature['activity_page0_kur'] = kurt
    feature['activity_page0_ske'] = skew
    #print feature['activity_page0_ske']
    feature['activity_page0_last'] = final_act(list(page_dict[0].values()))

    # print("test5")
    #page1
    feature['activity_page1_mean'] = np.mean(list(page_dict[1].values()))
    feature['activity_page1_std'] = np.var(list(page_dict[1].values()))
    feature['activity_page1_max'] = np.max(list((page_dict[1].values())))
    feature['activity_page1_min'] = np.min(list(page_dict[1].values()))
    [skew, kurt] = calc_stat(page_dict[1].values())
    feature['activity_page1_kur'] = kurt
    feature['activity_page1_ske'] = skew
    # print feature['activity_page0_ske']
    feature['activity_page1_last'] = final_act(list(page_dict[1].values()))

    # print("test6")
    #page2
    feature['activity_page2_mean'] = np.mean(list(page_dict[2].values()))
    feature['activity_page2_std'] = np.var(list(page_dict[2].values()))
    feature['activity_page2_max'] = np.max(list((page_dict[2].values())))
    feature['activity_page2_min'] = np.min(list(page_dict[2].values()))
    [skew, kurt] = calc_stat(page_dict[2].values())
    feature['activity_page2_kur'] = kurt
    feature['activity_page2_ske'] = skew
    # print feature['activity_page0_ske']
    feature['activity_page2_last'] = final_act(list(page_dict[2].values()))

    # print("test7")
    #page3
    feature['activity_page3_mean'] = np.mean(list(page_dict[3].values()))
    feature['activity_page3_std'] = np.var(list(page_dict[3].values()))
    feature['activity_page3_max'] = np.max(list((page_dict[3].values())))
    feature['activity_page3_min'] = np.min(list(page_dict[3].values()))
    [skew, kurt] = calc_stat(page_dict[3].values())
    feature['activity_page3_kur'] = kurt
    feature['activity_page3_ske'] = skew
    # print feature['activity_page0_ske']
    feature['activity_page3_last'] = final_act(list(page_dict[3].values()))

    # print("test8")
    # # page4
    feature['activity_page4_mean'] = np.mean(list(page_dict[4].values()))
    feature['activity_page4_std'] = np.var(list(page_dict[4].values()))
    feature['activity_page4_max'] = np.max(list((page_dict[4].values())))
    feature['activity_page4_min'] = np.min(list(page_dict[4].values()))
    [skew, kurt] = calc_stat(page_dict[4].values())
    feature['activity_page4_kur'] = kurt
    feature['activity_page4_ske'] = skew
    # print feature['activity_page0_ske']
    feature['activity_page4_last'] = final_act(list(page_dict[4].values()))
    Sactivityday=pd.Series(row['activity_day'])
    Sactivityday = Sactivityday.append(pd.Series([endday])).reset_index(drop=True)
    Sactivityday = Sactivityday.diff()
    Sactivityday = pd.Series(Sactivityday).dropna().reset_index(drop=True)
    feature["activity_daydiff_mean"] = Sactivityday.mean()
    feature["activity_daydiff_var"] = Sactivityday.var()
    feature["activity_daydiff_max"] = Sactivityday.max()
    feature["activity_daydiff_min"] = Sactivityday.min()
    feature["activity_daydiff_skew"] = Sactivityday.skew()
    feature["activity_daydiff_kurt"] = Sactivityday.kurt()
    list_con_activity=pd.Series(continuous(Sactivityday))
    feature["activity_con_count"] = list_con_activity.count()
    feature["activity_con_maxcount"] = list_con_activity.max()
    feature["activity_con_avgcount"] = list_con_activity.mean()
    feature["activity_con_varcount"] = list_con_activity.var()
    
    weekend_activity = row[(row.activity_day%7==0) | ((row.activity_day +1)%7==0)]
    feature["activity_weekendcount"]= len(weekend_activity)
    day_count=len(weekend_activity['activity_day'].drop_duplicates())
    feature['weekend_day_count'] =day_count
    feature['weekend_0_page_count'] = len(weekend_activity[(weekend_activity.user_id==feature['user_id']) & (weekend_activity.page==0)])
    feature['weekend_1_page_count'] = len(weekend_activity[(weekend_activity.user_id == feature['user_id']) & (weekend_activity.page == 1)])
    feature['weekend_2_page_count'] = len(weekend_activity[(weekend_activity.user_id == feature['user_id']) & (weekend_activity.page == 2)])
    feature['weekend_3_page_count'] = len(weekend_activity[(weekend_activity.user_id == feature['user_id']) & (weekend_activity.page == 3)])
    feature['weekend_4_page_count'] = len(weekend_activity[(weekend_activity.user_id == feature['user_id']) & (weekend_activity.page == 4)])
    if day_count>0:
        feature['weekend_mean_0_page_count'] = len(weekend_activity[(weekend_activity.user_id==feature['user_id']) & (weekend_activity.page==0)])/day_count
        feature['weekend_mean_1_page_count'] = len(weekend_activity[(weekend_activity.user_id == feature['user_id']) & (weekend_activity.page == 1)])/day_count
        feature['weekend_mean_2_page_count'] = len(weekend_activity[(weekend_activity.user_id == feature['user_id']) & (weekend_activity.page == 2)])/day_count
        feature['weekend_mean_3_page_count'] = len(weekend_activity[(weekend_activity.user_id == feature['user_id']) & (weekend_activity.page == 3)])/day_count
        feature['weekend_mean_4_page_count'] = len(weekend_activity[(weekend_activity.user_id == feature['user_id']) & (weekend_activity.page == 4)])/day_count
    else:
        feature['weekend_mean_0_page_count'] = 0
        feature['weekend_mean_1_page_count'] = 0
        feature['weekend_mean_2_page_count'] = 0
        feature['weekend_mean_3_page_count'] = 0
        feature['weekend_mean_4_page_count'] = 0


    feature['sum_page_activityVal_lastoneday'] = len(row[(row.user_id == feature['user_id'])  & (row.activity_day==endday)])
    feature['0_page_activityVal_lastoneday'] = len(row[(row.user_id == feature['user_id']) & (row.page == 0) & (row.activity_day==endday)])
    feature['1_page_activityVal_lastoneday'] = len(row[(row.user_id == feature['user_id']) & (row.page == 1) & (row.activity_day==endday)])
    feature['2_page_activityVal_lastoneday'] = len(row[(row.user_id == feature['user_id']) & (row.page == 2) & (row.activity_day==endday)])
    feature['3_page_activityVal_lastoneday'] = len(row[(row.user_id == feature['user_id']) & (row.page == 3) & (row.activity_day==endday)])
    feature['4_page_activityVal_lastoneday'] = len(row[(row.user_id == feature['user_id']) & (row.page == 4) & (row.activity_day==endday)])


    feature['sum_page_activityVal_lasttwoday'] = len(row[(row.user_id == feature['user_id'])  & (row.activity_day==endday-1)])
    feature['0_page_activityVal_lasttwoday'] = len(row[(row.user_id == feature['user_id']) & (row.page == 0) & (row.activity_day==endday-1)])
    feature['1_page_activityVal_lasttwoday'] = len(row[(row.user_id == feature['user_id']) & (row.page == 1) & (row.activity_day==endday-1)])
    feature['2_page_activityVal_lasttwoday'] = len(row[(row.user_id == feature['user_id']) & (row.page == 2) & (row.activity_day==endday-1)])
    feature['3_page_activityVal_lasttwoday'] = len(row[(row.user_id == feature['user_id']) & (row.page == 3) & (row.activity_day==endday-1)])
    feature['4_page_activityVal_lasttwoday'] = len(row[(row.user_id == feature['user_id']) & (row.page == 4) & (row.activity_day==endday-1)])

    feature['sum_page_activityVal_lastthrday'] = len(row[(row.user_id == feature['user_id'])  & (row.activity_day==endday-2)])
    feature['0_page_activityVal_lastthrday'] = len(row[(row.user_id == feature['user_id']) & (row.page == 0) & (row.activity_day==endday-2)])
    feature['1_page_activityVal_lastthrday'] = len(row[(row.user_id == feature['user_id']) & (row.page == 1) & (row.activity_day==endday-2)])
    feature['2_page_activityVal_lastthrday'] = len(row[(row.user_id == feature['user_id']) & (row.page == 2) & (row.activity_day==endday-2)])
    feature['3_page_activityVal_lastthrday'] = len(row[(row.user_id == feature['user_id']) & (row.page == 3) & (row.activity_day==endday-2)])
    feature['4_page_activityVal_lastthrday'] = len(row[(row.user_id == feature['user_id']) & (row.page == 4) & (row.activity_day==endday-2)])

    feature['sum_page_activityVal_lastfourday'] = len(row[(row.user_id == feature['user_id'])  & (row.activity_day==endday-3)])
    feature['0_page_activityVal_lastfourday'] = len(row[(row.user_id == feature['user_id']) & (row.page == 0) & (row.activity_day==endday-3)])
    feature['1_page_activityVal_lastfourday'] = len(row[(row.user_id == feature['user_id']) & (row.page == 1) & (row.activity_day==endday-3)])
    feature['2_page_activityVal_lastfourday'] = len(row[(row.user_id == feature['user_id']) & (row.page == 2) & (row.activity_day==endday-3)])
    feature['3_page_activityVal_lastfourday'] = len(row[(row.user_id == feature['user_id']) & (row.page == 3) & (row.activity_day==endday-3)])
    feature['4_page_activityVal_lasttfourday'] = len(row[(row.user_id == feature['user_id']) & (row.page == 4) & (row.activity_day==endday-3)])

    feature['sum_page_activityVal_lastfiveday'] = len(row[(row.user_id == feature['user_id'])  & (row.activity_day==endday-4)])
    feature['0_page_activityVal_lastfiveday'] = len(row[(row.user_id == feature['user_id']) & (row.page == 0) & (row.activity_day==endday-4)])
    feature['1_page_activityVal_lastfiveday'] = len(row[(row.user_id == feature['user_id']) & (row.page == 1) & (row.activity_day==endday-4)])
    feature['2_page_activityVal_lastfiveday'] = len(row[(row.user_id == feature['user_id']) & (row.page == 2) & (row.activity_day==endday-4)])
    feature['3_page_activityVal_lastfiveday'] = len(row[(row.user_id == feature['user_id']) & (row.page == 3) & (row.activity_day==endday-4)])
    feature['4_page_activityVal_lastfiveday'] = len(row[(row.user_id == feature['user_id']) & (row.page == 4) & (row.activity_day==endday-4)])

    feature['sum_page_activityVal_lastsixday'] = len(row[(row.user_id == feature['user_id'])  & (row.activity_day==endday-5)])
    feature['0_page_activityVal_lastsixday'] = len(row[(row.user_id == feature['user_id']) & (row.page == 0) & (row.activity_day==endday-5)])
    feature['1_page_activityVal_lastsixday'] = len(row[(row.user_id == feature['user_id']) & (row.page == 1) & (row.activity_day==endday-5)])
    feature['2_page_activityVal_lastsixday'] = len(row[(row.user_id == feature['user_id']) & (row.page == 2) & (row.activity_day==endday-5)])
    feature['3_page_activityVal_lastsixday'] = len(row[(row.user_id == feature['user_id']) & (row.page == 3) & (row.activity_day==endday-5)])
    feature['4_page_activityVal_lastsixday'] = len(row[(row.user_id == feature['user_id']) & (row.page == 4) & (row.activity_day==endday-5)])


    feature['sum_page_activityVal_lastsevenday'] = len(row[(row.user_id == feature['user_id'])  & (row.activity_day==endday-6)])
    feature['0_page_activityVal_lastsevenday'] = len(row[(row.user_id == feature['user_id']) & (row.page == 0) & (row.activity_day==endday-6)])
    feature['1_page_activityVal_lastsevenday'] = len(row[(row.user_id == feature['user_id']) & (row.page == 1) & (row.activity_day==endday-6)])
    feature['2_page_activityVal_lastsevenday'] = len(row[(row.user_id == feature['user_id']) & (row.page == 2) & (row.activity_day==endday-6)])
    feature['3_page_activityVal_lastsevenday'] = len(row[(row.user_id == feature['user_id']) & (row.page == 3) & (row.activity_day==endday-6)])
    feature['4_page_activityVal_lastsevenday'] = len(row[(row.user_id == feature['user_id']) & (row.page == 4) & (row.activity_day==endday-6)])
    # print('sad')
    return feature


def deal_feature(path,user_id):
    reg = pd.read_csv(path+register)
    cre = pd.read_csv(path+create)
    lau = pd.read_csv(path+launch)
    act = pd.read_csv(path+activity)
    feature = pd.DataFrame()
    feature['user_id'] = user_id



    act_feature = act.groupby('user_id',sort=True).apply(get_activity_feature)

    feature = pd.merge(feature,pd.DataFrame(act_feature),on='user_id',how='left')
    feature = feature.fillna(0)
    print('table activity is done')

    # cre['max_day'] = np.max(reg['register_day'])
    cre_feature = cre.groupby('user_id',sort=True).apply(get_create_feature)
    feature =  pd.merge(feature,pd.DataFrame(cre_feature),on='user_id',how='left')
    print('table CREATE feature is done')

    # reg['max_day'] = np.max(reg['register_day'])
    reg_feature = reg.groupby('user_id',sort=True).apply(get_register_feature)
    feature = pd.merge(feature,pd.DataFrame(reg_feature),on='user_id',how='left')
    print('table registre is done')

    # lau['max_day'] = np.max(reg['register_day'])
    lau_feature = lau.groupby('user_id',sort=True).apply(get_launch_feature)
    feature = pd.merge(feature,pd.DataFrame(lau_feature),on='user_id',how='left')
    print('table launch is done')

    # act['max_day'] = np.max(reg['register_day'])
    return feature


def get_data_feature():
    global endday
    global startday
    startday=1
    endday=16
    one_train_data = get_train_label(one_dataSet_train_path,one_dataSet_test_path)
    one_feature = deal_feature(one_dataSet_train_path,one_train_data['user_id'])
    one_feature['label'] = one_train_data['label']
    print("the first group feature is done")
    startday=8
    endday=23
    two_train_data = get_train_label(two_dataSet_train_path,two_dataSet_test_path)
    two_feature = deal_feature(two_dataSet_train_path,two_train_data['user_id'])
    two_feature['label'] = two_train_data['label']
    print("the second group feature is done")
    
    one_feature.to_csv(one_path,index=False)
    two_feature.to_csv(two_path,index=False)
    train_feature = pd.concat([one_feature,two_feature])
    train_feature.to_csv(train_path,index=False)
    print("trian data is save done")
    startday=15
    endday=30
    test_data = get_test(three_dataSet_train_path)
    test_feature = deal_feature(three_dataSet_train_path,test_data['user_id'])
    test_feature.to_csv(test_path,index=False)
    print("test save data done")

get_data_feature()
