import os
import numpy as np
import numpy.random as npr
import pandas as pd

from sys import path
path.append("../../")
from GomPlex import *

print('Loading data ...')

p, t = 0.1, 0.3
iter_tol = 30
diff_tol = 1e-5
ratio = 0.25
cv_folds = 3
model_path = 'best.pkl'
plot_error = False
metric = Metric('mse')

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
id_train = train['id'].tolist()
data_train = train.drop(['id'], axis=1).as_matrix()
X_train, y_train = data_train[:, :-2], data_train[:, -2:]
id_test = test['id'].tolist()
data_test = test.drop(['id'], axis=1).as_matrix()
X_test = data_test

print('Preprocessing data ...')

X_train[:, 0], X_test[:, 0] = np.log(1+X_train[:, 0]), np.log(1+X_test[:, 0])
y_train = np.log(1+y_train)
y_train = y_train[:, 0][:, None]+y_train[:, 1][:, None]*1j

print("Start training ...")

valid_size = int(X_train.shape[0]*t)
id_valid = np.random.choice(range(X_train.shape[0]), valid_size)
id_train = np.setdiff1d(range(X_train.shape[0]), id_valid)
X_train, y_train, X_valid, y_valid =\
    X_train[id_train], y_train[id_train], X_train[id_valid], y_train[id_valid]

print('  Gathered %d Training Examples.'%(X_train.shape[0]))
print('  Gathered %d Validation Examples.'%(X_valid.shape[0]))
print('  Gathered %d Testing Examples.'%(X_test.shape[0]))
print('  Done.')

while (True):
    print('# Training GomPlex')
    gp = GomPlex(npr.randint(np.log(X_train.shape[0])**2)+20, True)
    gp.fit(X_train, y_train, cost_type=metric.metric,
        iter_tol=iter_tol, diff_tol=diff_tol, cv_folds=cv_folds, plot=False)
    print('  Done.')
    print('# Choosing GomPlex Models')
    score = metric.eval(y_valid, *gp.predict(X_valid))
    print('  new score = %.8f'%(score))
    if(not os.path.exists(model_path)):
        gp.save(model_path)
    else:
        best_gp = GomPlex().load(model_path).fit(X_train, y_train)
        best_score = metric.eval(y_valid, *best_gp.predict(X_valid))
        print('  best score = %.8f'%(best_score))
        if(score < best_score):
            gp.save(model_path)
            backup_path = 'save_models/%s_%.6f.pkl'%(
                best_gp.hashed_name, best_score)
            best_gp.save(backup_path)
            print('  Found New Model!')
    
            print("Start prediction ...")
            y_pred = gp.predict(X_test)[0].ravel()
            
            print("Start write result ...")
            sub = pd.DataFrame()
            sub["id"] = id_test
            sub["formation_energy_ev_natom"] = np.exp(y_pred.real)-1
            sub["bandgap_energy_ev"] = np.exp(y_pred.imag)-1
            sub.to_csv(str(score)+"_.csv", index=False)
            
best_gp = GomPlex().load(model_path).fit(X_train, y_train)
y_pred = best_gp.predict(X_test)[0].ravel()
sub = pd.DataFrame()
sub["id"] = id_test
sub["formation_energy_ev_natom"] = np.exp(y_pred.real)-1
sub["bandgap_energy_ev"] = np.exp(y_pred.imag)-1
sub.to_csv(str(score)+"_.csv", index=False)








