import pandas as pd
import numpy as np
import gc
import stats as sts

"""
    代码懒得整理了。。。能用就行
    - action表统计特征
    1.总的，最后7,5,3,1天：action数，Page 0 1 2 3 4个数,action_type 0 1 2 3 4 5个数，收看vedio数，收看最多的那个vedio数，收看author数，收看次数最多的那个author数。
    2.1的比例特征(占对应时间段的总action数的比例)
    3.action , Page 0 1 2 3 4 , action_type 0 1 2 3 4 5 最后发生的那个离窗口结尾的天数。
    4.最长连续登陆天数，最长连续登陆的最后一天离滑窗末尾
    5.action最多的那一天的相对日期，最多使用次数
    6.user_id和author_id相同的频次，占总action比例
    7.最后7天是否有action(所有日志都算) , 最后三天是否有action , 最后一天是否有action.
    8.周末总次数，Page 0 1 2 3 4 , action_type 0 1 2 3 4 5在周末的次数
    9.Page 0 1 2 3 4 , action_type 0 1 2 3 4 5在周末的次数 占 周末总次数的比例
"""


# 得到序列特征
def get_seq_feature(seq, seq_name, user_id):
    # total 11 features
    if not seq:
        print('seq is empty!')
        return
    df = pd.DataFrame()
    # 一个值也可以
    df[seq_name + '_mean'] = [np.mean(seq)]
    df[seq_name + '_median'] = [np.median(seq)]
    df[seq_name + '_max'] = [np.max(seq)]
    df[seq_name + '_min'] = [np.min(seq)]
    df[seq_name + '_var'] = [np.var(seq)]
    df[seq_name + '_std'] = [np.std(seq)]
    # 需要最少两个值
    if len(seq) == 1:
        df[seq_name + '_upquantile'] = seq[0]
        df[seq_name + '_downquantile'] = 0
    else:
        df[seq_name + '_upquantile'] = [sts.quantile(seq, p=0.75)]
        df[seq_name + '_downquantile'] = [sts.quantile(seq, p=0.25)]
    # 平均值不能为空
    if np.mean(seq) != 0:
        df[seq_name + '_discrete'] = [np.std(seq) / np.mean(seq)]
    else:
        df[seq_name + '_discrete'] = [np.NaN]
    # 可能无法计算
    try:
        df[seq_name + 'skew'] = [sts.skewness(seq)]
    except:
        df[seq_name + 'skew'] = [np.NaN]
    try:
        df[seq_name + 'kurt'] = [sts.kurtosis(seq)]
    except:
        df[seq_name + 'kurt'] = [np.NaN]
    df['user_id'] = [user_id]
    return df


# 统计action
def day_count(df, count_df, feature_name):
    count_df = count_df.groupby('user_id').count()
    count_df = count_df[['day']]
    count_df = count_df.reset_index('user_id')
    count_df.rename(columns={'day': feature_name}, inplace=True)
    count_df = count_df[['user_id', feature_name]]
    df = pd.merge(df, count_df, how='left', on='user_id')
    df[feature_name] = df[feature_name].fillna(0)
    df[feature_name] = df[feature_name].astype(int)
    return df


# 统计 page 和 type | 比例
def other_count(df, others, feature_name, other_name):
    others = pd.DataFrame(others.groupby(['user_id', feature_name])[feature_name].count())
    others = others.reset_index('user_id')
    others.rename(columns={feature_name: feature_name + '_count'}, inplace=True)
    others = others.reset_index(feature_name)
    others = others.pivot('user_id', feature_name, feature_name + '_count').reset_index('user_id')
    columns = others.columns.tolist()
    percent_lists = []
    for c in columns:
        if c is not 'user_id':
            others.rename(columns={c: feature_name + '_' + str(c) + '_' + other_name}, inplace=True)
            percent_lists.append(feature_name + '_' + str(c) + '_' + other_name)
    df = pd.merge(df, others, how='left', on='user_id')
    # extract percentage feature
    for c in percent_lists:
        df[c + '_percent'] = df[c] / df[other_name + '_action_count']
    return df


# 统计 video_id , author_id
def video_count(df, action, feature_name, other_name):
    def getVideo(l):
        return len(np.unique(l))

    temp = action.groupby(['user_id', feature_name])[feature_name].count().reset_index('user_id')
    temp.rename(columns={feature_name: feature_name + '_count' + '_' + other_name}, inplace=True)
    temp = temp.reset_index(feature_name)
    temp_num = temp.copy()
    temp = temp.groupby(['user_id']).agg('max').reset_index('user_id')[
        ['user_id', feature_name + '_count' + '_' + other_name]]
    temp_num = temp_num.groupby('user_id').agg(getVideo).reset_index('user_id')[
        ['user_id', feature_name]]
    temp_num.rename(columns={feature_name: feature_name + '_' + other_name}, inplace=True)
    # 最多的那个video收看了多少次
    df = pd.merge(df, temp, how='left', on='user_id')
    # 一共收看了多少个video
    df = pd.merge(df, temp_num, how='left', on='user_id')
    df[feature_name + '_count' + '_' + other_name + '_percent'] = df[feature_name + '_count' + '_' + other_name] / df[
        other_name + '_action_count']
    df[feature_name + '_' + other_name + '_percent'] = df[feature_name + '_' + other_name] / df[
        other_name + '_action_count']
    return df


# action , Page 0 1 2 3 4 , action_type 0 1 2 3 4 5 最后一次相对日期
def last_day(df, action, start, other_name):
    # all action
    all_action = action.groupby(['user_id'])['day'].agg('max')
    all_action = all_action.reset_index('user_id')
    all_action.rename(columns={'day': other_name + '_all_action_last_day'}, inplace=True)
    all_action[other_name + '_all_action_last_day'] = all_action[other_name + '_all_action_last_day'] - start
    df = pd.merge(df, all_action, how='left', on='user_id')
    # Page 0 1 2 3 4 | action_type 0 1 2 3 4 5
    all_feature = ['page', 'action_type']
    for i in range(len(all_feature)):
        temp = action.groupby(['user_id', all_feature[i]])['day'].agg('max')
        temp = temp.reset_index(['user_id', all_feature[i]])
        temp['day'] = temp['day'] - start
        temp = temp.pivot('user_id', all_feature[i], 'day').reset_index('user_id')
        for c in temp.columns:
            if c is not 'user_id':
                temp.rename(columns={c: other_name + '_' + all_feature[i] + '_' + str(c) + '_last_day'}, inplace=True)
        df = pd.merge(df, temp, how='left', on='user_id')
    return df


# 最长连续登陆天数，最长登陆天数最后一天相对日期，action最多那一天的相对日期,最多单天action数
def continous_day(df, action, other_name):
    def longest_day(seq):
        count = 1
        longest = 1
        last_one = seq[0]
        for i in range(len(seq)):
            if i + 1 < len(seq):
                if seq[i] + 1 == seq[i + 1]:
                    count += 1
                    if count > longest:
                        longest = count
                        last_one = seq[i + 1]
                else:
                    count = 1
        return [longest, last_one]

    def most_action(df):
        seq_day = df.day
        seq_day_count = df.day_count
        max_day_count = 0
        max_day = 0
        for x, y in zip(seq_day, seq_day_count):
            if y > max_day_count:
                max_day = x
                max_day_count = y
        return max_day

    page = action.groupby(['user_id', 'day'])['day'].count()
    page = page.reset_index(['user_id'])
    page.rename(columns={'day': 'day_count'}, inplace=True)
    page = page.reset_index('day')
    temp_df = pd.DataFrame({
        'user_id': np.unique(page.user_id).tolist()
    })
    longest = []
    longest_last = []
    most_action_date = []
    for user in temp_df.user_id:
        piece_df = page[page.user_id == user]
        longest.append(longest_day(piece_df.day.tolist())[0])
        longest_last.append(longest_day(piece_df.day.tolist())[1])
        most_action_date.append(most_action(piece_df))
    temp_df[other_name + '_longest_action_day'] = longest
    temp_df[other_name + '_longest_action_day_last'] = longest_last
    temp_df[other_name + '_max_action_day'] = most_action_date
    df = pd.merge(df, temp_df, how='left', on='user_id')
    page = page.groupby('user_id')['day_count'].agg('max')
    page = page.reset_index('user_id')
    page.rename(columns={'day_count': other_name + '_max_action_one_day'}, inplace=True)
    df = pd.merge(df, page, how='left', on='user_id')
    return df


# 最后一天是否有拍摄video的行为
def whether_video(df, video, end):
    video = video[video.day == end]
    video = video.drop_duplicates()
    video.loc[video.day > 0, 'day'] = 1
    video.rename(columns={'day': 'whether_last_video'}, inplace=True)
    df = pd.merge(df, video, how='left', on='user_id')
    df['whether_last_video'] = df['whether_last_video'].fillna(0).astype(int)
    return df


# user_id 和 author_id 相同的 action
def same_author(df, action, other_name):
    temp = action[action.user_id == action.author_id].groupby('user_id').count().reset_index('user_id')[
        ['user_id', 'day']]
    temp.rename(columns={'day': other_name + '_same_author_count'}, inplace=True)
    df = pd.merge(df, temp, how='left', on='user_id')
    df[other_name + '_same_author_count'] = df[other_name + '_same_author_count'].fillna(0).astype(int)
    df[other_name + '_same_author_count_percent'] = df[other_name + '_same_author_count'] / df[
        other_name + '_action_count']
    return df


# 最后七天，三天，两天，一天是否有action
def whether_action(df, action, other_name):
    temp = action.groupby('user_id').count().reset_index('user_id')[['user_id', 'day']]
    temp.rename(columns={'day': other_name + '_whether_active'}, inplace=True)
    temp.loc[temp[other_name + '_whether_active'] > 0, other_name + '_whether_active'] = 1
    df = pd.merge(df, temp, how='left', on='user_id')
    df[other_name + '_whether_active'] = df[other_name + '_whether_active'].fillna(0).astype(int)
    return df


def extract_action(df, action, start, end):
    # action 总次数, 最后 7 5 3 1
    df = day_count(df, action, 'all_action_count')
    df = day_count(df, action[(action.day >= end - 6) & (action.day <= end)], 'last_seven_action_count')
    df = day_count(df, action[(action.day >= end - 4) & (action.day <= end)], 'last_five_action_count')
    df = day_count(df, action[(action.day >= end - 2) & (action.day <= end)], 'last_three_action_count')
    df = day_count(df, action[action.day == end], 'last_one_action_count')

    # 周末总次数 , 周末的 Page 0 1 2 3 4 个数 , action_type 0 1 2 3 4 5 个数
    df = day_count(df, action[(action.day % 7 == 0) | ((action.day + 1) % 7 == 0)], 'all_weekend_count')
    for i in range(5):
        df = day_count(df, action[((action.day % 7 == 0) | ((action.day + 1) % 7 == 0)) & (action.page == i)], 'all_weekend_count_page_%d'%i)
    for i in range(6):
        df = day_count(df, action[((action.day % 7 == 0) | ((action.day + 1) % 7 == 0)) & (action.action_type == i)],'all_weekend_count_type_%d' % i)
    # 周末的 Page 0 1 2 3 4 个数 , action_type 0 1 2 3 4 5 个数 分别占 周末总次数的比例
    for i in range(5):
        df['all_weekend_count_page_%d_percent'%i] = df['all_weekend_count_page_%d'%i] / df['all_weekend_count']
    for i in range(6):
        df['all_weekend_count_type_%d_percent'%i] = df['all_weekend_count_type_%d'%i] / df['all_weekend_count']


    # 总的Page 0 1 2 3 4 个数,最后7,5,3,1 | 总的action_type 0 1 2 3 4 5 个数，最后7,5,3,1
    all_type = ['page', 'action_type']
    for i in range(2):
        df = other_count(df, action, all_type[i], 'all')
        df = other_count(df, action[(action.day >= end - 6) & (action.day <= end)], all_type[i], 'last_seven')
        df = other_count(df, action[(action.day >= end - 4) & (action.day <= end)], all_type[i], 'last_five')
        df = other_count(df, action[(action.day >= end - 2) & (action.day <= end)], all_type[i], 'last_three')
        df = other_count(df, action[action.day == end], all_type[i], 'last_one')

    # video 看了几个，看的最多的那个video有多少次action ，author 同 video
    all_type = ['author_id', 'video_id']
    for i in range(2):
        df = video_count(df, action, all_type[i], 'all')
        df = video_count(df, action[(action.day >= end - 6) & (action.day <= end)], all_type[i], 'last_seven')
        df = video_count(df, action[(action.day >= end - 4) & (action.day <= end)], all_type[i], 'last_five')
        df = video_count(df, action[(action.day >= end - 2) & (action.day <= end)], all_type[i], 'last_three')
        df = video_count(df, action[action.day == end], all_type[i], 'last_one')

    # action , Page 0 1 2 3 4 , action_type 0 1 2 3 4 5 最后一次相对日期
    df = last_day(df, action, start_date_list[0], 'all')

    # 总的，最后7,5,3,1 ： 最长连续登陆天数，最长登陆天数最后一天相对日期，action最多那一天的相对日期, 最多单天action数
    df = continous_day(df, action, 'all')
    df = continous_day(df, action[(action.day >= end - 6) & (action.day <= end)], 'last_seven')
    df = continous_day(df, action[(action.day >= end - 4) & (action.day <= end)], 'last_five')
    df = continous_day(df, action[(action.day >= end - 2) & (action.day <= end)], 'last_three')
    df = continous_day(df, action[action.day == end], 'last_one')

    # user_id 和 author_id 相同的 count | percentage
    df = same_author(df, action, 'all')
    df = same_author(df, action[(action.day >= end - 6) & (action.day <= end)], 'last_seven')
    df = same_author(df, action[(action.day >= end - 4) & (action.day <= end)], 'last_five')
    df = same_author(df, action[(action.day >= end - 2) & (action.day <= end)], 'last_three')
    df = same_author(df, action[action.day == end], 'last_one')

    # 最后七天，三天，两天，一天是否有action
    df = whether_action(df, action, 'all')
    df = whether_action(df, action[(action.day >= end - 6) & (action.day <= end)], 'last_seven')
    df = whether_action(df, action[(action.day >= end - 4) & (action.day <= end)], 'last_five')
    df = whether_action(df, action[(action.day >= end - 2) & (action.day <= end)], 'last_three')
    df = whether_action(df, action[action.day == end], 'last_one')
    return df

def clear_variable():
    global launch_day_df
    global launch_day_gap_df
    launch_day_df = pd.DataFrame()
    launch_day_gap_df = pd.DataFrame()



if __name__ == '__main__':
    # variable
    launch_day_df = pd.DataFrame()
    launch_day_gap_df = pd.DataFrame()
    # parameter
    Num_dataSet = 3
    start_date_list = [1, 8, 15]
    end_date_list = [16, 23, 30]

    # extract
    for i in range(Num_dataSet):
        print('Dataset %d' % (i + 1))
        action = pd.read_csv('splited_date/action_%d_%d.csv' % (start_date_list[i], end_date_list[i]), index_col=0)
        video = pd.read_csv('splited_date/video_%d_%d.csv' % (start_date_list[i], end_date_list[i]), index_col=0)
        df = pd.read_csv('splited_date/df_%d_%d.csv' % (start_date_list[i], end_date_list[i]), index_col=0)
        df = extract_action(df, action, start_date_list[i], end_date_list[i])
        df = whether_video(df, video, end_date_list[i])
        df.to_csv('Most_action_features_%d_%d.csv' % (start_date_list[i],end_date_list[i]))
