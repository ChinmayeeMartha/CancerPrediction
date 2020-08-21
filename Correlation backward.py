import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings("ignore")


from sklearn.metrics import accuracy_score as acc
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs

# Read data
data = pd.read_csv('tumor.csv')
'''data = data.iloc[:,1:-1]
label_encoder = LabelEncoder()
data.iloc[:,0] = label_encoder.fit_transform(data.iloc[:,0]).astype('float64')
result = pd.DataFrame()
result['diagnosis'] = data.iloc[:,0]'''


# Train/test split
X_train, X_test, y_train, y_test = train_test_split(data.values[:,2:11],
    data.values[:,1],
    test_size=0.25,
    random_state=42)
    

y_train = y_train.ravel()
y_test = y_test.ravel()

print('Training dataset shape:', X_train.shape, y_train.shape)
print('Testing dataset shape:', X_test.shape, y_test.shape)
corr = data.corr()
columns = np.full((corr.shape[0],), True, dtype=bool)
for i in range(corr.shape[1]):
    for j in range(i+1, corr.shape[0]):
        if corr.iloc[i,j] >= 0.6:
            if columns[j]:
                columns[j] = False
selected_columns = data.columns[columns]
data = data[selected_columns]
selected_columns2 = selected_columns[1:].values
s=data.corr(method ='pearson')
print(s)
print(selected_columns2)
print(data[selected_columns].describe())
# Build RF classifier to use in feature selection
clf = RandomForestClassifier(n_estimators=1, n_jobs=-1)
svc=SVC()
# Build step backward feature selection
sfs1 = sfs(clf, 
          k_features=(1,9), 
          forward=False, 
          floating=False, 
          scoring='accuracy',
          )
sfs2 = sfs(svc, 
          k_features=(1,9), 
          forward=False, 
          floating=False, 
          scoring='accuracy',
          )
sfs1 = sfs1.fit(X_train, y_train)
sfs2 = sfs2.fit(X_train, y_train)
feat_cols = list(sfs1.k_feature_idx_)
print(feat_cols)
feat_cols2 = list(sfs2.k_feature_idx_)
print(feat_cols2)
fig1=plot_sfs(sfs1.get_metric_dict(),kind='std_dev')
plt.ylabel('accuracy')
plt.xlim([11,2])
plt.ylim([0.2,0.4])
plt.show()
fig2=plot_sfs(sfs2.get_metric_dict(),kind='std_dev')
plt.ylabel('accuracy')
plt.xlim([11,2])
plt.ylim([0.2,0.6])
plt.show()
svc=SVC()
clf = RandomForestClassifier(n_estimators=1, n_jobs=-1)
all_accuracies = cross_val_score(estimator=svc, X=X_train, y=y_train, cv=3)
print(all_accuracies)
print("SVM accuracy",all_accuracies.mean())
# The default kernel used by SVC is the gaussian kernel
svc.fit(X_train, y_train)
svmprediction = svc.predict(X_test)
cm = confusion_matrix(y_test, svmprediction)
print(cm)
sum = 0





for i in range(cm.shape[0]):
    sum += cm[i][i]    
accuracy = sum/X_test.shape[0]
print("SVM accuracy:",accuracy)
specificity=cm[0,0]/(cm[0,0]+cm[0,1])
print("SVM specificity:",specificity)
sensitivity=cm[1,1]/(cm[1,0]+cm[1,1])
print("SVM sensitivity",sensitivity)
all_accuracies = cross_val_score(estimator=clf, X=X_train, y=y_train, cv=3)
print(all_accuracies)
print("RF accuracy",all_accuracies.mean())
# The default kernel used by SVC is the gaussian kernel
clf.fit(X_train, y_train)
clfprediction = clf.predict(X_test)
cm = confusion_matrix(y_test, clfprediction)
print(cm)
sum = 0





for i in range(cm.shape[0]):
    sum += cm[i][i]    
accuracy = sum/X_test.shape[0]
print("RF accuracy:",accuracy)
specificity=cm[0,0]/(cm[0,0]+cm[0,1])
print("RF specificity:",specificity)
sensitivity=cm[1,1]/(cm[1,0]+cm[1,1])
print("RF sensitivity",sensitivity)


clf = RandomForestClassifier(n_estimators=1, random_state=10, max_depth=4)
clf.fit(X_train[:, feat_cols], y_train)
svc=SVC()
svc.fit(X_train[:, feat_cols], y_train)
y_train_pred = clf.predict(X_train[:, feat_cols])
print('Training accuracy on selected features: %.3f' % acc(y_train, y_train_pred))

y_test_pred = clf.predict(X_test[:, feat_cols])
print('Testing accuracy on selected features: %.3f' % acc(y_test, y_test_pred))
# Build full model on ALL features, for comparison
clf = RandomForestClassifier(n_estimators=12, random_state=42, max_depth=4)
clf.fit(X_train, y_train)

y_train_pred = clf.predict(X_train)
print('Training accuracy on all features: %.3f' % acc(y_train, y_train_pred))

y_test_pred = clf.predict(X_test)
print('Testing accuracy on all features: %.3f' % acc(y_test, y_test_pred))

svc = SVC()
svc.fit(X_train, y_train)

y_train_pred = clf.predict(X_train)
print('Training accuracy on all features: %.3f' % acc(y_train, y_train_pred))

y_test_pred = clf.predict(X_test)
print('Testing accuracy on all features: %.3f' % acc(y_test, y_test_pred))
y_train_pred = svc.predict(X_train)
print('Training accuracy on all features: %.3f' % acc(y_train, y_train_pred))

y_test_pred = svc.predict(X_test)
print('Testing accuracy on all features: %.3f' % acc(y_test, y_test_pred))



