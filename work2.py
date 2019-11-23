import scipy.io as sio
import pandas as pd
data=sio.loadmat('data_label.mat')
df_data=pd.DataFrame(data['data'])
df_label=pd.DataFrame(data['label'])

print(df_data)
print(df_label)