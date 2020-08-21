from __future__ import print_function
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
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
import operator

from sklearn.metrics import accuracy_score as acc
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs

class FeatureSelection:

    def __init__(self, csv, num_feature_select):

        # self.cols = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13', 'x14', 'y' ]
        # self.cols = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 'y']
        # self.cols = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 'y']
        #self.cols = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 'y']
        #self.num_cols = len(self.cols)
        self.information_gain = {}  # Information gain for all features numbered 0 - (n - 1)
        self.num_feature_select = num_feature_select  # Number of top features to select
        self.top_n_features = []  # Top n features

        self.discrete_features = [0, 3, 4, 5, 7, 8, 10, 11]    # Features having discrete Values
        # self.discrete_features = [0, 2, 3, 5, 6, 8, 9, 11, 13, 14, 16, 18, 19]  # Features having discrete Values
        self.csv_data = pd.read_csv('wpbc.csv')
        # self.X = self.csv_data.iloc[:, 0:20];
        # self.Y = self.csv_data.iloc[:, 20: 21].values.reshape(-1,)
        self.X = self.csv_data.iloc[:, 2:32]
        self.Y = self.csv_data.iloc[:, 1]

        # self.gain = {}
        # print(self.Y.values.reshape(-1,))

    def exp_IG(self):
        x = self.X.values
        y = self.Y.values

        def _entropy(values):
            counts = np.bincount(values).astype('float64')
            probs = counts[np.nonzero(counts)] / float(len(values))
            # print(1 - probs, probs)
            return np.sum(probs * np.exp(1 - probs))

        def ig(feature, y):
            feature_set_indices = np.nonzero(feature)
            feature_not_set_indices = [i for i in feature_range if i not in feature_set_indices[0]]
            entropy_x_set = _entropy(y[feature_set_indices])
            entropy_x_not_set = _entropy(y[feature_not_set_indices])

            return entropy_before - (((len(feature_set_indices) / float(feature_size)) * entropy_x_set)
                                     + ((len(feature_not_set_indices) / float(feature_size)) * entropy_x_not_set))

        feature_size = x.shape[0]
        feature_range = range(0, feature_size)
        # print(feature_size)
        # print(feature_range)
        entropy_before = _entropy(y)
        # print(entropy_before)
        information_gain_scores = []
        # print(x.T.shape)
        for feature in x.T:
            # print(feature)
            information_gain_scores.append(ig(feature, y))
        # print(information_gain_scores)
        info_gain = {}

        for i in range(self.X.shape[1]):
            info_gain[str(i)] = information_gain_scores[i]

        info_gain = sorted(info_gain.items(), key=operator.itemgetter(1), reverse=True)

        for i in range(self.X.shape[1]):
            if i < self.num_feature_select:
                self.top_n_features.append(int(info_gain[i][0]))
            self.information_gain[info_gain[i][0]] = info_gain[i][1]

        # return information_gain_scores, []

    def mutual_info_calculator(self):
        information_gain = []
        information_gain.append(mutual_info_regression(self.X, self.Y, discrete_features=self.discrete_features))

        info_gain = {}

        for i in range(self.X.shape[1]):
            info_gain[str(i)] = information_gain[0][i]

        info_gain = sorted(info_gain.items(), key=operator.itemgetter(1), reverse=True)

        for i in range(self.X.shape[1]):
            if i < self.num_feature_select:
                self.top_n_features.append(int(info_gain[i][0]))
            self.information_gain[info_gain[i][0]] = info_gain[i][1]


# p = FeatureSelection("GermanData.csv", 10)
p = FeatureSelection("wpbc.csv", 32)

p.exp_IG()
print(p.information_gain)
print(p.top_n_features)
X_train, X_test, y_train, y_test = train_test_split(p.X.values,p.Y.values,random_state=100, test_size = 0.2)

clf = RandomForestClassifier(n_estimators=1, n_jobs=-1)
svc=SVC()
# Build step forward feature selection
sfs1 = sfs(clf,
           k_features=29,
           forward=True,
           floating=False,
           verbose=2,
           scoring='accuracy')
sfs2 = sfs(svc,
           k_features=29,
           forward=True,
           floating=False,
           verbose=2,
           scoring='accuracy')

# Perform SFFS
sfs1 = sfs1.fit(X_train, y_train)
sfs2 = sfs2.fit(X_train, y_train)
# Which features?
feat_cols = list(sfs1.k_feature_idx_)
feat_cols2 =list(sfs2.k_feature_idx_)
print(feat_cols)
print(feat_cols2)
fig1=plot_sfs(sfs1.get_metric_dict(),kind='std_dev')
plt.ylabel('accuracy')
plt.xlim([2,32])
plt.ylim([0.8,1])
plt.show()

fig2=plot_sfs(sfs2.get_metric_dict(),kind='std_dev')
plt.ylabel('accuracy')
plt.xlim([2,32])
plt.ylim([0.8,1])
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
# Build full model with selected features
clf = RandomForestClassifier(n_estimators=1, random_state=10, max_depth=4)
clf.fit(X_train[:, feat_cols], y_train)
svc = SVC()
svc.fit(X_train[:, feat_cols], y_train)

y_train_pred = clf.predict(X_train[:, feat_cols])
print('Training accuracy on selected features: %.3f' % acc(y_train, y_train_pred))
y_train_pred = svc.predict(X_train[:, feat_cols])
print('Training accuracy on selected features: %.3f' % acc(y_train, y_train_pred))

y_test_pred = clf.predict(X_test[:, feat_cols])
print('Testing accuracy on selected features: %.3f' % acc(y_test, y_test_pred))
# Build full model on ALL features, for comparison
clf = RandomForestClassifier(n_estimators=12, random_state=42, max_depth=4)
clf.fit(X_train, y_train)
y_test_pred = svc.predict(X_test[:, feat_cols])
print('Testing accuracy on selected features: %.3f' % acc(y_test, y_test_pred))
# Build full model on ALL features, for comparison
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




