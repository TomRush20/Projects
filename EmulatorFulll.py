import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score, recall_score, precision_score, f1_score
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
from mlxtend.classifier import StackingCVClassifier

#Constants
RANDOM_SEED = 20
FOLDS = 5
TARGET = 'Malware'


data_frame = pd.read_csv('EmuFull.csv')
data_frame.drop('Package', inplace=True, axis=1)
print(data_frame.shape)

#Splitting Timeframe
data_frame['FirstModDate'] = pd.to_datetime(data_frame['FirstModDate'], format='%m/%d/%Y', errors='coerce')
data_frame['LastModDate'] = pd.to_datetime(data_frame['LastModDate'], format='%m/%d/%Y', errors='coerce')

data_frame['FirstModDate_year'] = data_frame['FirstModDate'].dt.year
data_frame['FirstModDate_month'] = data_frame['FirstModDate'].dt.month
data_frame['FirstModDate_day'] = data_frame['FirstModDate'].dt.day

data_frame['LastModDate_year'] = data_frame['LastModDate'].dt.year
data_frame['LastModDate_month'] = data_frame['LastModDate'].dt.month
data_frame['LastModDate_day'] = data_frame['LastModDate'].dt.day

data_frame.drop('FirstModDate', inplace=True, axis=1)
data_frame.drop('LastModDate', inplace=True, axis=1)

print(data_frame.head)

#label encoding the string columns
le = preprocessing.LabelEncoder()

data_frame['sha256'] = le.fit_transform(data_frame['sha256'])
data_frame['MalFamily'] = le.fit_transform(data_frame['MalFamily'])

classes, count = np.unique(data_frame['Malware'], return_counts=True)

print(le.fit_transform(classes), classes)
data_frame = data_frame.replace(classes, le.fit_transform(classes))

#Dropping the missing values
data_frame = data_frame.replace(['None,?'], np.NaN, regex=True)
print(f'Total Missing Values: {sum(list(data_frame.isna().sum()))}')
data_frame.dropna(inplace=True)

for c in data_frame.columns:
    data_frame[c] = pd.to_numeric(data_frame[c])

#Shuffle the dataset
data_frame = data_frame.sample(n=len(data_frame), random_state=20)
data_frame = data_frame.reset_index(drop=True)
print(data_frame.head)

#classifier prep
X = data_frame.drop(columns=TARGET)
Y = data_frame[TARGET]

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=RANDOM_SEED)

#classifier parameters
rf_params = {
    'max_depth': 15,
    'min_samples_leaf': 8,
    'random_state': RANDOM_SEED,
}
dt_params = {
    'max_depth': 5,
    'random_state': RANDOM_SEED,
}
cb_params = {
    'task_type': 'CPU',
    'logging_level': 'Silent',
    'random_state': RANDOM_SEED,
}
et_params = {
    'criterion': 'entropy',
    'max_features': .55,
    'min_samples_leaf': 8,
    'min_samples_split': 4,
    'n_estimators': 100,
    'random_state': RANDOM_SEED,
}
classifiers = {
    "Random Forest": RandomForestClassifier(**rf_params),
    "Decision Tree": DecisionTreeClassifier(**dt_params),
    "CatBoost": CatBoostClassifier(**cb_params),
    "ExtraTrees": ExtraTreesClassifier(**et_params),
}

#Training the classifiers
scores = {}
print('Training Classifiers')
for name, clf in classifiers.items():
    scores[name] = cross_val_score(clf, x_train, y_train, cv=FOLDS, scoring='accuracy')
    print(f'{name} accuracy: {scores[name].mean():.5f} (+/- {scores[name].std() * 2:.5f})')
    clf.fit(x_train, y_train)

fig = plt.figure(figsize=(15,6))
sns.boxplot(data=pd.DataFrame(scores), showmeans=True)
plt.title('Classification Accuracy')
plt.xlabel('Classifier')
plt.ylabel('Accuracy')
plt.show()

#Stacked Classifier
meta_clf = LogisticRegression(random_state=RANDOM_SEED)
stacked_clf = StackingCVClassifier(classifiers=list(classifiers.values()), meta_classifier=meta_clf, cv=FOLDS, use_probas=True,
                                   random_state=RANDOM_SEED)
scores = cross_val_score(stacked_clf, x_train, y_train, cv=FOLDS, scoring='accuracy')
print(f'Stacked Accuracy {scores.mean():.5f} (+/- {scores.std() * 2:.5f})')
stacked_clf.fit(x_train, y_train)
classifiers['Stacked'] = stacked_clf

model_performance = pd.DataFrame(columns=['Accuracy', 'Recall', 'Precision', 'F1-Score'])

#Accuracy Score
for name, clf in classifiers.items():
    #y_pred = clf.predict(x_test)
    y_pred = cross_val_predict(clf, X, Y, cv=FOLDS)
    accuracy = accuracy_score(Y, y_pred)
    recall = recall_score(Y, y_pred)
    precision = precision_score(Y, y_pred)
    f1 = f1_score(Y, y_pred)
    model_performance.loc[name] = [accuracy, recall, precision, f1]

model_performance.style.background_gradient(cmap='coolwarm')
print(model_performance)

#Confusion Matrix
fig, axes = plt.subplots(3, 3, figsize=(15, 13))
for i, (name, clf) in enumerate(classifiers.items()):
    if name == 'Stacked':
        continue
    #y_pred = clf.predict(x_test)
    y_pred = cross_val_predict(clf, X, Y, cv=FOLDS)
    sns.heatmap(confusion_matrix(Y, y_pred), annot=True, fmt='d', ax=axes[i // 3, i % 3])
    axes[i // 3, i % 3].set_title(name)

#y_pred = classifiers['Stacked'].predict(x_test)
y_pred = cross_val_predict(classifiers['Stacked'], X, Y, cv=FOLDS)
sns.heatmap(confusion_matrix(Y, y_pred), annot=True, fmt='d', ax=axes[2,1])
axes[2,1].set_title('Stacked')

fig.delaxes(axes[2,0])
fig.delaxes(axes[2,2])
plt.show()

#ROC Curve
line_styles = ['-', '--', '-.', ':']
markers = ['o', 'v', '^', '<', '>', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X']

fig, ax = plt.subplots(figsize=(10,10))
for idx, (name, clf) in enumerate(classifiers.items()):
    y_pred = clf.predict_proba(x_test)[:,1]
    fpr, tpr, threshold = roc_curve(y_test, y_pred)
    ax.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc_score(y_test, y_pred):.5f})', marker=markers[idx % len(markers)],
            linestyle=line_styles[idx % len(line_styles)])
ax.plot([0,1], [0,1], 'k--')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve')
ax.legend(loc='lower right')
plt.show()

#Classification Report
for name, clf in classifiers.items():
    y_pred = clf.predict(x_test)
    print(f'{name} Classification Report: \n{classification_report(y_test, y_pred)}')