import numpy as np
import numpy.random as npr
import pandas as pd
import gc

from sys import path
path.append("../../")
from GomPlex import *

print('Loading data ...')

p, t = 0.1, 0.3
iter_tol = 30
ratio = 0.3
cv_folds = 3
score_rerun = 20
model_path = 'best.pkl'
plot_error = False
metric = Metric('mae')

train = pd.read_csv('train2016.csv')
prop = pd.read_csv('prop2016.csv')

print('Preprocessing data ...')

missing = (prop.isnull().sum(axis=0)/prop.shape[0]).reset_index()
missing.columns = ['column', 'missing_p']
drop_cols = missing.ix[missing['missing_p']>0.8]['column'].values.tolist()

cnt_cols = [col for col in prop.columns if 'cnt' in col]
prop[cnt_cols] = prop[cnt_cols].fillna(value=0)
yr_cols = [col for col in prop.columns if 'year' in col]
filter_cols = []
for col in cnt_cols+yr_cols:
    if(np.unique(prop[col]).shape[0]<50):
        filter_cols.append(col)
        prop[col] = prop[col].astype('category')
    else:
        prop[col] -= prop[col].min()
prop[yr_cols] = prop[yr_cols].fillna(value=0)
df_data = pd.get_dummies(prop[filter_cols])

cat_cols = [col for col in prop.columns if 'id' in col or 'flag' in col]
cat_cols.remove('parcelid')
prop[cat_cols] = prop[cat_cols].fillna(value=0)
filter_cat_cols = []
for col in cat_cols:
    if(np.unique(prop[col]).shape[0]<50):
        filter_cat_cols.append(col)
        prop[col] = prop[col].astype('category')
df_data = pd.concat([df_data, pd.get_dummies(prop[filter_cat_cols])], axis=1)

num_cols = [col for col in prop.columns if col not in cat_cols+filter_cols]
df_data = pd.concat([df_data, prop[num_cols]], axis=1)

dates = pd.to_datetime(pd.Series(train['transactiondate'].tolist()))
df_data['date'] = ((dates-dates.min()).astype('timedelta64[D]').astype(int))
df_data = df_data.fillna(value=0)

print('Generating training data ...')

df_train = train.merge(df_data, how='left', on='parcelid')
y_train = df_train['logerror'].values
y_train = np.sign(y_train)*(np.abs(y_train)**p)
y_train_r = np.maximum(0, y_train)
y_train_i = np.maximum(0, -y_train)
y_train = y_train_r+1j*y_train_i
X_train = df_train.drop(['parcelid', 'logerror', 'transactiondate', 'propertyzoningdesc', 'propertycountylandusecode', 'censustractandblock',
'rawcensustractandblock'], axis=1)
columns = X_train.columns
print(X_train.shape, y_train.shape)


print("Start training ...")

split = int(X_train.shape[0]*(1-t))
X_train, y_train, X_valid, y_valid =\
    X_train[:split], y_train[:split], X_train[split:], y_train[split:]
X_train = X_train.as_matrix()
X_valid = X_valid.as_matrix()
    
print("Generating testing data ...")

sample = pd.read_csv('sample.csv')
sample['parcelid'] = sample['ParcelId']
df_test = sample.merge(df_data, on='parcelid', how='left')
X_test = df_test[columns]
X_test = X_test.as_matrix()
result = pd.read_csv('sample.csv')

del df_train, df_test, df_data; gc.collect()

print('  Gathered %d Training Examples.'%(X_train.shape[0]))
print('  Gathered %d Testing Examples.'%(X_test.shape[0]))
print('  Done.')
print('# Training GomPlex')
gp = GomPlex(npr.randint(int(np.log(X_train.shape[0]))*3)+8, True)
gp.fit(X_train, y_train, cost_type=metric.metric,
    iter_tol=iter_tol, cv_folds=cv_folds, plot=plot_error)
print('  Done.')
print('# Choosing GomPlex Models')
score = metric.eval(y_valid, *gp.predict(X_valid))
print('  new score = %.3f'%(score))
if(not os.path.exists(model_path)):
    gp.save(model_path)
else:
    best_gp = GomPlex().load(model_path).fit(X_train, y_train)
    best_score = metric.eval(y_valid, *best_gp.predict(X_valid))
    print('  best score = %.3f'%(best_score))
    if(score > best_score):
        gp.save(model_path)
        backup_path = 'save_models/%s_%.6f.pkl'%(gp.hashed_name, best_score)
        gp.save(backup_path)
        print('  Found New Model!')

        print("Start prediction ...")
        test_dates = [288, 319, 349, 653, 684, 714]
        for i, test_date in enumerate(test_dates):
            X_test[:, -1] = test_date
            y_test = gp.predict(X_test)[0].ravel()
            result[result.columns[i+1]] = y_test.real-y_test.imag
        
        print("Start write result ...")
        result.to_csv(gp.hashed_name+'.csv', index=False, float_format='%.6f')








