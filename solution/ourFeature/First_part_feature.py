"""
    author: yang yiqing
    comment: 前三个日志的特征提取
    date: 2018年06月16日16:31:49

"""

import pandas as pd
import numpy as np
import stats as sts

"""
    - 提取注册特征(register)
    1.相对注册日期，在时间窗之前注册的为-1
    2.注册类型
    3.设备类型
    
"""


def extract_register(df, register, start, end):
    print('提取注册特征...')
    df = pd.merge(df, register, how='left', on='user_id')
    # 相对日期特征 在时间窗之前注册的为-1
    df['register_day'] = df['day'] - start + 1
    df['register_day'] = df['register_day'].fillna(-1)
    df['register_day'] = df['register_day'].astype(int)
    temp = df.copy()
    all_register = pd.read_csv('data/register.csv')
    # 注册类型 设备类型
    temp = pd.merge(temp[['user_id']], all_register[['user_id', 'register_type', 'device_type']], how='left',
                    on='user_id')
    temp.to_csv('feature_set/%d_%d_register_feature.csv' % (start, end))
    df = df.drop_duplicates()
    df[['user_id', 'label']].to_csv('feature_set/%d_%d_df.csv' % (start, end))


"""
    - 提取启动特征(launch)
    1.launch天数序列特征 : 所有序列特征
    2.launch_diff序列(加入end day取diff) : 所有序列特征 ， 最后一个和倒数第二个值(没有则为0)
    3.launch 总天数,最后 7 5 3 1天的次数
    5.launch 第一次启动到时间窗末尾
    4.launch percent ： 启动总天数 / 第一次启动到时间窗末尾
    6.launch 连续天数序列： 所有序列特征
    7.周末的launch总次数(unique)
    8.周末的Launch总数占总launch数比
"""


def extract_launch(df, launch, start, end):
    print('提取launch特征..')
    def get_feature(line):
        global launch_day_df, launch_day_gap_df, launch_day_continuous_df
        user_id = line['user_id'].tolist()[0]
        seq = sorted(line['day'].tolist())
        length = end - seq[0] + 1
        seq = [x - start + 1 for x in seq]
        # launch day 作为序列
        each_launch_day = get_seq_feature(seq, 'launch_day', user_id)
        launch_day_df = pd.concat([launch_day_df, each_launch_day])
        seq = pd.Series(seq + [end - start + 1])
        seq_diff = seq.diff().dropna().reset_index(drop=True).tolist()
        # launch day diff 作为序列
        each_launch_gap_day = get_seq_feature(seq_diff, 'launch_gap_day', user_id)
        # 取倒数第一和第倒数第二个
        each_launch_gap_day['launch_day_diff_last_one'] = [seq_diff[-1]]
        each_launch_gap_day['launch_day_diff_last_two'] = [seq_diff[-2]] if len(seq_diff) > 1 else [0]
        # duration / percent
        each_launch_gap_day['launch_day_duration'] = [length]
        each_launch_gap_day['launch_day_percent'] = (len(seq)-1)/length
        # 周末总次数
        weekend_launch = [len([day for day in np.unique(seq[:-1]) if day % 7 == 0 or (day + 1) % 7 == 0])]
        each_launch_gap_day['launch_weekend_day_count'] = weekend_launch
        each_launch_gap_day['launch_weekend_day_count_percent'] = [weekend_launch[0] / (len(seq) - 1)]
        launch_day_gap_df = pd.concat([launch_day_gap_df, each_launch_gap_day])
        # launch continuous day 序列
        seq = seq.tolist()[:-1]
        continuous_launch_day = get_seq_feature(get_continuous(seq), 'continuous_launch_day', user_id)
        launch_day_continuous_df = pd.concat([launch_day_continuous_df, continuous_launch_day])


    launch.groupby('user_id').apply(lambda x: get_feature(x))
    global launch_day_gap_df, launch_day_df, launch_day_continuous_df
    launch_day_gap_df = launch_day_gap_df.drop_duplicates()
    launch_day_df = launch_day_df.drop_duplicates()
    launch_day_continuous_df = launch_day_continuous_df.drop_duplicates()
    launch_day_df[1:].to_csv('feature_set/%d_%d_launch_day.csv' % (start, end), float_format='%.2f')
    launch_day_gap_df[1:].to_csv('feature_set/%d_%d_launch_day_diff.csv' % (start, end), float_format='%.2f')
    launch_day_continuous_df[1:].to_csv('feature_set/%d_%d_launch_day_continuous.csv' % (start, end), float_format='%.2f')
    # launch 总次数 最后 7 5 3 1 天次数
    df = day_count(df, launch.groupby('user_id').count(), 'all_launch_count')
    df = day_count(df, launch[(launch.day >= end - 6) & (launch.day <= end)].groupby('user_id').count(), 'last_seven_launch_count')
    df = day_count(df, launch[(launch.day >= end - 4) & (launch.day <= end)].groupby('user_id').count(),'last_five_launch_count')
    df = day_count(df, launch[(launch.day >= end - 2) & (launch.day <= end)].groupby('user_id').count(),'last_three_launch_count')
    df = day_count(df, launch[launch.day == end].groupby('user_id').count(),'last_one_launch_count')
    clear_variable()

    # 周末总次数 / 周末发生天数 无意义，launch次数一天最多只有一次
    df.drop('label',axis=1).to_csv('feature_set/%d_%d_launch_normal.csv' % (start, end), float_format='%.2f')

"""
    - 提取拍摄视频特征(create)
    1.create天数序列特征 : 所有序列特征
    2.create_diff序列(取unique,加入end day取diff) : 所有序列特征 ， 最后一个和倒数第二个值(没有则为0)
    3.create 总天数,最后 7 5 3 1天的次数
    4.第一次拍摄到时间窗末尾 
    5.create percent ： 拍摄总天数(unique) / 第一次拍摄到时间窗末尾
    6.create day连续序列: 所有序列特征
    7.create count序列 : 所有序列特征
    8.create count diff序列：所有序列特征 最后一个和倒数第一个值(没有则为0) 加和
    9.周末的 create 总数
    10.发生create的周末天数(unique)
    10.周末create总次数 占比 周末create天数(unique)
"""
def extract_create(df,create,start,end):
    print('提取create特征..')
    def get_feature(line):
        global create_day_df, create_day_gap_df, create_day_continuous_df, create_day_count_df, create_day_count_diff_df
        user_id = line['user_id'].tolist()[0]
        seq = sorted(line['day'].tolist())
        duration = end - seq[0] + 1
        seq = [x - start + 1 for x in seq]
        # create day 作为序列
        each_create_day = get_seq_feature(seq, 'create_day', user_id)
        create_day_df = pd.concat([create_day_df, each_create_day])
        # create day count 序列
        # print(seq)
        seq_count = [seq.count(day) for day in np.unique(seq)]
        # print(seq_count)
        each_create_count = get_seq_feature(seq_count, 'create_day_count', user_id)
        create_day_count_df = pd.concat([create_day_count_df, each_create_count])
        # create day count diff 序列
        seq_count_diff = pd.Series(seq_count).diff().dropna().reset_index(drop=True).tolist()
        seq_count_diff = [0] if not seq_count_diff else seq_count_diff
        each_create_count_diff = get_seq_feature(seq_count_diff, 'create_day_count_diff', user_id)
        # 取倒数第一和第倒数第二个，加和
        each_create_count_diff['create_day_count_diff_last_one'] = [seq_count_diff[-1]] if len(
            seq_count_diff) > 0 else [0]
        each_create_count_diff['create_day_count_diff_last_two'] = [seq_count_diff[-2]] if len(
            seq_count_diff) > 1 else [0]
        each_create_count_diff['create_day_count_diff_sum'] = [np.sum(seq_count_diff)]
        create_day_count_diff_df = pd.concat([create_day_count_diff_df, each_create_count_diff])
        seq = pd.Series(np.unique(seq).tolist() + [end - start + 1])
        seq_diff = seq.diff().dropna().reset_index(drop=True).tolist()
        # create day diff 作为序列
        each_create_gap_day = get_seq_feature(seq_diff, 'create_gap_day', user_id)
        # 取倒数第一和第倒数第二个
        each_create_gap_day['create_day_diff_last_one'] = [seq_diff[-1]]
        each_create_gap_day['create_day_diff_last_two'] = [seq_diff[-2]] if len(seq_diff) > 1 else [0]
        # duration 以及 percent
        each_create_gap_day['create_day_duration'] = [duration]
        each_create_gap_day['create_day_percent'] = [(len(seq)-1)/duration]
        # weekend count / percent
        each_create_gap_day['create_weekend_day_count'] = [len([day for day in np.unique(seq[:-1]) if day % 7 == 0 or (day + 1) % 7 == 0])]
        each_create_gap_day['create_weekend_day_all_count'] = [len([day for day in seq[:-1] if day % 7 == 0 or (day + 1) % 7 == 0])]
        each_create_gap_day['create_weekend_day_count_percent'] = each_create_gap_day['create_weekend_day_count'] / each_create_gap_day['create_weekend_day_all_count']
        create_day_gap_df = pd.concat([create_day_gap_df, each_create_gap_day])
        # create continuous day 序列
        seq = seq.tolist()[:-1]
        continuous_create_day = get_seq_feature(get_continuous(seq), 'continuous_create_day', user_id)
        create_day_continuous_df = pd.concat([create_day_continuous_df, continuous_create_day])


    create.groupby('user_id').apply(lambda x: get_feature(x))
    global create_day_gap_df, create_day_df, create_day_continuous_df, create_day_count_df, create_day_count_diff_df
    # save feature file
    create_day_df = create_day_df.drop_duplicates()
    create_day_gap_df = create_day_gap_df.drop_duplicates()
    create_day_continuous_df = create_day_continuous_df.drop_duplicates()
    create_day_count_df = create_day_count_df.drop_duplicates()
    create_day_count_diff_df = create_day_count_diff_df.drop_duplicates()
    create_day_df.to_csv('feature_set/%d_%d_create_day.csv' % (start, end), float_format='%.2f')
    create_day_gap_df.to_csv('feature_set/%d_%d_create_day_diff.csv' % (start, end), float_format='%.2f')
    create_day_continuous_df.to_csv('feature_set/%d_%d_create_day_continuous.csv' % (start, end), float_format='%.2f')
    create_day_count_df.to_csv('feature_set/%d_%d_create_day_count.csv' % (start, end), float_format='%.2f')
    create_day_count_diff_df.to_csv('feature_set/%d_%d_create_day_count_diff.csv' % (start, end), float_format='%.2f')
    # create 总次数 最后 7 5 3 1 天次数
    df = day_count(df, create.groupby('user_id').count(), 'all_create_count')
    df = day_count(df, create[(create.day >= end - 6) & (create.day <= end)].groupby('user_id').count(),'last_seven_create_count')
    df = day_count(df, create[(create.day >= end - 4) & (create.day <= end)].groupby('user_id').count(),'last_five_create_count')
    df = day_count(df, create[(create.day >= end - 2) & (create.day <= end)].groupby('user_id').count(),'last_three_create_count')
    df = day_count(df, create[create.day == end].groupby('user_id').count(), 'last_one_create_count')
    # save features file
    df.drop('label', axis=1).to_csv('feature_set/%d_%d_create_normal.csv' % (start, end), float_format='%.2f')
    clear_variable()
    print(df.columns)







"""
    tools函数
    - get_seq_feature : 得到这个序列的所有序列特征
      均值 中位数 最大 最小 方差 标准差 上下四分位数，变异稀疏，峰度，偏度 (无法计算填np.NaN)
    - day_count : 次数统计
    - get_continuous : 得到连续天数序列
"""


# 得到序列特征
def get_seq_feature(seq, seq_name, user_id):
    if not seq:
        print('seq is empty! : %s'%seq_name)
        return
    df = pd.DataFrame()
    df[seq_name + '_mean'] = [np.mean(seq)]
    df[seq_name + '_median'] = [np.median(seq)]
    df[seq_name + '_max'] = [np.max(seq)]
    df[seq_name + '_min'] = [np.min(seq)]
    df[seq_name + '_var'] = [np.var(seq)]
    df[seq_name + '_std'] = [np.std(seq)]
    if np.mean(seq) != 0:
        df[seq_name + '_discrete'] = [np.std(seq) / np.mean(seq)]
    else:
        df[seq_name + '_discrete'] = [np.NaN]
    try:
        df[seq_name + '_skew'] = [sts.skewness(seq)]
    except:
        df[seq_name + '_skew'] = [np.NaN]
    try:
        df[seq_name + '_kurt'] = [sts.kurtosis(seq)]
    except:
        df[seq_name + '_kurt'] = [np.NaN]
    df['user_id'] = [user_id]
    return df

# 次数统计
def day_count(df, count_df, feature_name):
    count_df = count_df.reset_index('user_id')
    count_df.rename(columns={'day': feature_name}, inplace=True)
    df = pd.merge(df, count_df, how='left', on='user_id')
    df[feature_name] = df[feature_name].fillna(0)
    df[feature_name] = df[feature_name].astype(int)
    return df

# 连续登陆天数序列
def get_continuous(seq):
    seq = pd.Series(seq).diff().dropna().reset_index(drop=True).tolist()
    con_seq, temp = [], 1
    while seq:
        while seq and seq.pop(0) == 1:
            temp += 1
        con_seq.append(temp) if temp > 1 else None
        temp = 1
    return con_seq if con_seq else [0]

# day序列转count序列
def get_count(seq):
    return

# clear global variable
def clear_variable():
    global launch_day_df,launch_day_gap_df,launch_day_continuous_df,create_day_df,create_day_gap_df,create_day_continuous_df\
        ,create_day_count_df,create_day_count_diff_df
    launch_day_df, launch_day_gap_df, launch_day_continuous_df, create_day_df, create_day_gap_df, create_day_continuous_df\
        , create_day_count_df, create_day_count_diff_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()\
        , pd.DataFrame(), pd.DataFrame(), pd.DataFrame()


if __name__ == '__main__':
    # global
    launch_day_df = pd.DataFrame()
    launch_day_gap_df = pd.DataFrame()
    launch_day_continuous_df = pd.DataFrame()
    create_day_df = pd.DataFrame()
    create_day_gap_df = pd.DataFrame()
    create_day_continuous_df = pd.DataFrame()
    create_day_count_df = pd.DataFrame()
    create_day_count_diff_df = pd.DataFrame()
    # parameter 修改时间窗需要修改下面的参数
    Num_dataSet = 3
    start_date_list = [1, 8, 15]
    end_date_list = [16, 23, 30]
    # extract
    for i in range(Num_dataSet):
        print('DataSet %d (%d ~ %d)' % (i + 1, start_date_list[i], end_date_list[i]))
        action = pd.read_csv('splited_date/action_%d_%d.csv' % (start_date_list[i], end_date_list[i]), index_col=0)
        launch = pd.read_csv('splited_date/launch_%d_%d.csv' % (start_date_list[i], end_date_list[i]), index_col=0)
        register = pd.read_csv('splited_date/register_%d_%d.csv' % (start_date_list[i], end_date_list[i]), index_col=0)
        create = pd.read_csv('splited_date/video_%d_%d.csv' % (start_date_list[i], end_date_list[i]), index_col=0)
        df = pd.read_csv('splited_date/df_%d_%d.csv' % (start_date_list[i], end_date_list[i]), index_col=0)
        # 每次提取一个DataSet(时间窗)的特征
        extract_register(df, register, start_date_list[i], end_date_list[i])
        extract_launch(df, launch, start_date_list[i], end_date_list[i])
        extract_create(df, create, start_date_list[i], end_date_list[i])
