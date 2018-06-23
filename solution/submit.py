# -*- coding: utf-8 -*-
# day31~day37真实活跃用户的个数大约为23727个
import pandas as pd
from datetime import datetime

result = pd.read_csv('../result/lgb_result.csv')
active_user_id = result.sort_values(by='result', ascending=False)
# active_user_id = active_user_id.head(23800)
active_user_id = result[result['result'] >= 0.42]
# active_user_id = result.sort_values(by='result', axis=0, ascending=False).iloc[0:23760,:]
# print('threshold:',active_user_id.iloc[-1,1])
print(len(active_user_id))

del active_user_id['result']
active_user_id.to_csv('../submission/submit_result_{}.txt'.format(datetime.now().strftime('%m%d_%H%M')), index=False, header=False)
