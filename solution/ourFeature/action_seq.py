import pandas as pd
import numpy as np
import gc
import stats as sts


# 得到序列特征
def get_seq_feature(seq, seq_name, user_id):
    # total 11 features
    if not seq:
        print('seq is empty!')
        return
    df = pd.DataFrame()
    df[seq_name + '_mean'] = [np.mean(seq)]
    df[seq_name + '_median'] = [np.median(seq)]
    df[seq_name + '_max'] = [np.max(seq)]
    df[seq_name + '_min'] = [np.min(seq)]
    df[seq_name + '_var'] = [np.var(seq)]
    df[seq_name + '_std'] = [np.std(seq)]
    if len(seq) == 1:
        df[seq_name + '_upquantile'] = seq[0]
        df[seq_name + '_downquantile'] = 0
    else:
        df[seq_name + '_upquantile'] = [sts.quantile(seq, p=0.75)]
        df[seq_name + '_downquantile'] = [sts.quantile(seq, p=0.25)]
    if np.mean(seq) != 0: df[seq_name + '_discrete'] = [np.std(seq) / np.mean(seq)]
    else: df[seq_name + '_discrete'] = [np.NaN]
    try: df[seq_name + 'skew'] = [sts.skewness(seq)]
    except: df[seq_name + 'skew'] = [np.NaN]
    try: df[seq_name + 'kurt'] = [sts.kurtosis(seq)]
    except: df[seq_name + 'kurt'] = [np.NaN]
    df['user_id'] = [user_id]
    return df


# lambda 引用的函数
def get_feature(line, other_name, start):
    global launch_day_df
    global launch_day_gap_df
    user_id = line['user_id'].tolist()[0]
    seq = sorted(line['day'].tolist())
    seq = [x - start for x in seq]
    # launch day 作为序列
    each_launch_day = get_seq_feature(seq, other_name, user_id)
    launch_day_df = pd.concat([launch_day_df, each_launch_day])
    # launch day gap 作为序列
    seq = np.unique(seq).tolist()
    seq_gap = []
    if len(seq) < 2:
        seq_gap = [0]
    else:
        for i in range(1, len(seq)):
            seq_gap.append(seq[i] - seq[i - 1])

    each_launch_gap_day = get_seq_feature(seq_gap, other_name + '_gap', user_id)
    launch_day_gap_df = pd.concat([launch_day_gap_df, each_launch_gap_day])


def get_every_day_seq_feature(line, other_name, start):
    global launch_day_df
    global launch_day_gap_df
    user_id = line['user_id']
    seq = sorted(line.values.tolist()[1:])
    seq = pd.Series(seq)
    seq = seq.fillna(0).tolist()
    # day 作为序列
    each_launch_day = get_seq_feature(seq, other_name, user_id)
    launch_day_df = pd.concat([launch_day_df, each_launch_day])
    # launch day gap 作为序列
    seq = np.unique(seq).tolist()
    seq_gap = []
    if len(seq) < 2:
        seq_gap = [0]
    else:
        for i in range(1, len(seq)):
            seq_gap.append(seq[i] - seq[i - 1])

    each_launch_gap_day = get_seq_feature(seq_gap, other_name + '_gap', user_id)
    launch_day_gap_df = pd.concat([launch_day_gap_df, each_launch_gap_day])


# action , page 0 1 2 3 4 , action_type 0 1 2 3 4 5 , user_id = author_id 发生的天数
def action_seq(df, action, start, end):
    print('提取action序列特征...')
    # 总 action 发生在哪一天的序列 和 间隔序列 特征
    action.groupby('user_id').apply(lambda x: get_feature(x, 'all_action', start))
    df = pd.merge(df, launch_day_df, how='left', on='user_id')
    df = pd.merge(df, launch_day_gap_df, how='left', on='user_id')
    clear_variable()
    df = df.drop_duplicates()
    df.to_csv('feature_set/%d_%d_action_seq.csv' % (start, end))


def page_seq(df, action, start, end):
    for i in range(5):
        print('提取page_%d序列特征...' % i)
        action[action.page == i].groupby('user_id').apply(lambda x: get_feature(x, 'page_%d' % i, start))
        df = pd.merge(df, launch_day_df, how='left', on='user_id')
        df = pd.merge(df, launch_day_gap_df, how='left', on='user_id')
        clear_variable()
    df = df.drop_duplicates()
    df.to_csv('feature_set/%d_%d_page_seq.csv' % (start, end))


def action_type_seq(df, action, start, end):
    for i in range(6):
        print('提取action_type_%d序列特征...' % i)
        action[action.action_type == i].groupby('user_id').apply(lambda x: get_feature(x, 'action_type_%d' % i, start))
        df = pd.merge(df, launch_day_df, how='left', on='user_id')
        df = pd.merge(df, launch_day_gap_df, how='left', on='user_id')
        clear_variable()
    df = df.drop_duplicates()
    df.to_csv('feature_set/%d_%d_action_type_seq.csv' % (start, end))


def user_author_seq(df, action, start, end):
    print('提取user_author序列特征')
    action[action.user_id == action.author_id].groupby('user_id').apply(
        lambda x: get_feature(x, 'same_user_author', start))
    df = pd.merge(df, launch_day_df, how='left', on='user_id')
    df = pd.merge(df, launch_day_gap_df, how='left', on='user_id')
    df = df.drop_duplicates()
    clear_variable()
    df.to_csv('feature_set/%d_%d_user_author_seq.csv' % (start, end))

# 每一天的值构成的序列提取特征
def action_every_day_seq(df, action, start, end):
    print('提取用户每天action数序列特征')
    action.groupby(['user_id', 'day']).count().reset_index().pivot('user_id', 'day', 'page')\
        .reset_index().apply(lambda x: get_every_day_seq_feature(x, 'action_everyday', start), axis=1)
    df = pd.merge(df, launch_day_df, how='left', on='user_id')
    df = pd.merge(df, launch_day_gap_df, how='left', on='user_id')
    df = df.drop_duplicates()
    clear_variable()
    df.to_csv('feature_set/%d_%d_action_everyday_seq.csv' % (start, end))


def page_every_day_seq(df, action, start, end):
    for i in range(5):
        print('提取每天page%d数序列特征...' % i)
        action[action.page == i].groupby(['user_id', 'day']).count().reset_index().pivot('user_id', 'day','page')\
            .reset_index().apply(lambda x: get_every_day_seq_feature(x, 'page%d_everyday' % i, start), axis=1)
        df = pd.merge(df, launch_day_df, how='left', on='user_id')
        df = pd.merge(df, launch_day_gap_df, how='left', on='user_id')
        df = df.drop_duplicates()
        clear_variable()
    df = df.drop_duplicates()
    df.to_csv('feature_set/%d_%d_page_everyday_seq.csv' % (start, end))

def action_type_every_day_seq(df, action, start, end):
    for i in range(6):
        print('提取每天action_type%d数序列特征...' % i)
        action[action.action_type == i].groupby(['user_id', 'day']).count().reset_index().pivot('user_id', 'day','page')\
            .reset_index().apply(lambda x: get_every_day_seq_feature(x, 'action_type_%d_everyday' % i, start), axis=1)
        df = pd.merge(df, launch_day_df, how='left', on='user_id')
        df = pd.merge(df, launch_day_gap_df, how='left', on='user_id')
        df = df.drop_duplicates()
        clear_variable()
    df = df.drop_duplicates()
    df.to_csv('feature_set/%d_%d_action_type_everyday_seq.csv' % (start, end))

def user_author_every_day_seq(df, action, start, end):
    print('提取用户每天user_author数序列特征')
    action[action.user_id == action.author_id].groupby(['user_id', 'day']).count().reset_index().pivot('user_id', 'day', 'page')\
        .reset_index().apply(lambda x: get_every_day_seq_feature(x, 'user_author_everyday', start), axis=1)
    df = pd.merge(df, launch_day_df, how='left', on='user_id')
    df = pd.merge(df, launch_day_gap_df, how='left', on='user_id')
    df = df.drop_duplicates()
    clear_variable()
    df.to_csv('feature_set/%d_%d_user_author_everyday_seq.csv' % (start, end))


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
        df = pd.read_csv('splited_date/df_%d_%d.csv' % (start_date_list[i], end_date_list[i]), index_col=0)
        df.pop('label')
        # 下面都是各个行为发生的那天构成的序列(以及该序列的gap序列)特征
        action_seq(df,action,start_date_list[i],end_date_list[i])
        page_seq(df,action,start_date_list[i],end_date_list[i])
        action_type_seq(df,action,start_date_list[i],end_date_list[i])
        user_author_seq(df,action,start_date_list[i],end_date_list[i])

        # 下面的加上作用不大
        # action_every_day_seq(df, action, start_date_list[i], end_date_list[i])
        # page_every_day_seq(df, action, start_date_list[i], end_date_list[i])
        # action_type_every_day_seq(df, action, start_date_list[i], end_date_list[i])
        # user_author_every_day_seq(df, action, start_date_list[i], end_date_list[i])


