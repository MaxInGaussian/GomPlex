import numpy as np
import numpy.random as npr
import pandas as pd
import pickle
import gc

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.cross_validation import cross_val_score

print('Loading data ...')

p, t = 0.1, 0.3
nb_epoch = 1000
batch_size = 100
cv_folds = 3
model_path = 'best_keras.pkl'

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

prop['taxdelinquencyflag'] = pd.get_dummies(prop['taxdelinquencyflag'])['Y']

cat_cols = [col for col in prop.columns if 'id' in col]
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
y_train = np.hstack(np.maximum(0, y_train)[:, None],
    np.maximum(0, -y_train)[:, None])
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

def save(regressor, path=model_path):
    with open(path, "wb") as save_f:
        pickle.dump(regressor, save_f, pickle.HIGHEST_PROTOCOL)

def load(path=model_path):
    with open(path, "rb") as load_f:
        return pickle.load(load_f)

def eval(p, y):
    return np.abs(p[:, 0]-p[:, 1]-y[:, 0]+y[:, 1]).mean()

def model():
    model = Sequential()
    model.add(Dense(128, input_dim=len(columns), init='normal'))
    model.add(Activation('relu'))
    model.add(Dropout(0.15))
    model.add(Dense(128, init='normal'))
    model.add(Activation('relu'))
    model.add(Dropout(0.15))
    model.add(Dense(2, init='normal'))
    model.compile(loss='mae', optimizer='adam')
    return model

while (True):
    print('# Training Keras')
    kr = KerasRegressor(build_fn=model,
        nb_epoch=nb_epoch, batch_size=batch_size, verbose=1)
    scores = cross_val_score(kr, X_train, y_train, cv=cv_folds)
    print('  Done.')
    print('# Choosing GomPlex Models')
    print("msle = %4.2f std = %4.2f" % (scores.mean(),scores.std()))
    p_valid = kr.model.predict(X_valid)
    score = eval(p_valid, y_valid)
    print('  new score = %.3f'%(score))
    if(not os.path.exists(model_path)):
        save(kr)
    else:
        best_kr = load().fit(X_train, y_train)
        p_valid = kr.model.predict(X_valid)
        best_score = eval(p_valid, y_valid)
        print('  best score = %.3f'%(best_score))
        if(score > best_score):
            save(kr)
            backup_path = 'save_models/%.6f.pkl'%(best_score)
            save(best_kr, backup_path)
            print('  Found New Model!')
    
            print("Start prediction ...")
            test_dates = [288, 319, 349, 653, 684, 714]
            for i, test_date in enumerate(test_dates):
                X_test[:, -1] = test_date
                y_test = kr.predict(X_test)
                result[result.columns[i+1]] = y_test[:, 0]-y_test[:, 1]
            
            print("Start write result ...")
            result.to_csv('%.6f.csv'%(score), index=False, float_format='%.6f')








