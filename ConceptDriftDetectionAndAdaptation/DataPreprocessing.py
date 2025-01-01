import pandas as pd
import warnings
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")


# Using Label encoder to transform string features to numerical features
def encoding(df_to_process):
    cat_features = [x for x in df_to_process.columns if df_to_process[x].dtype == "object"]
    le = LabelEncoder()
    for col in cat_features:
        if col in df_to_process.columns:
            i = df_to_process.columns.get_loc(col)
            df_to_process.iloc[:, i] = le.fit_transform(df_to_process.iloc[:, i].astype(str))
    return df_to_process


# Assign column names
col_names = ["duration", "protocol_type", "service", "flag", "src_bytes",
             "dst_bytes", "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
             "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
             "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
             "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
             "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
             "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
             "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
             "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
             "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label", "difficulty"]

# Read the original training and test sets
df1 = pd.read_csv("NSL-KDD-Dataset\\KDDTrain+.txt", header=None, names=col_names)
df2 = pd.read_csv("NSL-KDD-Dataset\\KDDTest+.txt", header=None, names=col_names)

# display the dataset
# print(df1)

# "normal" label is set to 0, all attack labels are set to 1

df1.drop(['difficulty'], axis=1, inplace=True)
df2.drop(['difficulty'], axis=1, inplace=True)

df1['label'][df1['label'] == 'normal'] = 0
df1['label'][df1['label'] != 0] = 1
df2['label'][df2['label'] == 'normal'] = 0
df2['label'][df2['label'] != 0] = 1

df1 = encoding(df1)
df2 = encoding(df2)

df = pd.concat([df1, df2], ignore_index=True)
df1.to_csv('NSL-KDD-Dataset\\NSL_KDD_binary_train.csv', index=0)
df2.to_csv('NSL-KDD-Dataset\\NSL_KDD_binary_test.csv', index=0)
df.to_csv('NSL-KDD-Dataset\\NSL_KDD_binary(train+test).csv', index=False)
