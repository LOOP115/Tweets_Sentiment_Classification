import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import load_digits
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
import re
from data_cleaning import *






# read data
train_data = pd.read_csv("Train.csv", sep=',')
test_data = pd.read_csv("Test.csv", sep=',')
# print(train_data)

#separating instance and label for Train
X_train_raw = [x[0] for x in train_data[['text']].values]
Y_train = [x[0] for x in train_data[['sentiment']].values]
X_test_raw = [x[0] for x in test_data[['text']].values]
# print(type(X_train_raw))
# print(X_test_raw)


# 1. data cleaning (optional)



X_train_need_to_clean = pd.DataFrame(X_train_raw)
X_test_need_to_clean = pd.DataFrame(X_test_raw)

# remove url, # and @
# X_train_need_to_clean.replace("\b*https?:\S*", '', regex=True, inplace=True)
X_train_need_to_clean.replace("\b*@\S*", '', regex=True, inplace=True)
X_train_need_to_clean.replace("\b*#\S*", '', regex=True, inplace=True)

X_test_need_to_clean.replace("\b*https?:\S*", '', regex=True, inplace=True)
X_test_need_to_clean.replace("\b*@\S*", '', regex=True, inplace=True)
X_test_need_to_clean.replace("\b*#\S*", '', regex=True, inplace=True)

# print(X_train_need_to_clean)

for i in range(X_train_need_to_clean.shape[0]):
    # X_train_need_to_clean.loc[i, 0] = remove_whitespace(X_train_need_to_clean.loc[i, 0])
    # X_train_need_to_clean.loc[i, 0] = expand_contractions(X_train_need_to_clean.loc[i, 0])
    # X_train_need_to_clean.loc[i, 0] = remove_whitespace(X_train_need_to_clean.loc[i, 0])
    X_train_need_to_clean.loc[i, 0] = ' '. join(text_preprocessing(X_train_need_to_clean.loc[i, 0], remove_html=False))

print(X_train_need_to_clean)

X_train_clean = [x[0] for x in X_train_need_to_clean[[0]].values]
X_test_clean = [x[0] for x in X_test_need_to_clean[[0]].values]







# 2. vectorization (transformation)

# # bag of words
# # countvectorizer
# BoW_vectorizer = CountVectorizer(analyzer='word', ngram_range=(2,2))
# X_train_BoW = BoW_vectorizer.fit_transform(X_train_clean)
# X_test_BoW = BoW_vectorizer.transform(X_test_clean)

# # TFIDF
# tfidf_vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(2,2))
# X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_raw)
# X_test_tfidf = tfidf_vectorizer.transform(X_test_raw)
#
# print("Train feature space size (using TFIDF):",X_train_tfidf.shape)
# print("Test feature space size (using TFIDF):",X_test_tfidf.shape)


# # save bow
# output_dict = BoW_vectorizer.vocabulary_
# output_pd = pd.DataFrame(list(output_dict.items()),columns = ['word','count'])
# output_pd.T.to_csv('BoW-vocab.csv',index=False)



# # 3. feature selection
# ##### want to see the filtered features in excel####
# X_train_new = SelectKBest(chi2,k=5000).fit_transform(X_train_BoW,Y_train)
# # print(X_train_new[0].toarray())
# # print(type(X_train_new[0]))

# # save bow
# output_dict = BoW_vectorizer.vocabulary_
# output_pd = pd.DataFrame(list(output_dict.items()),columns = ['word','count'])
# output_pd.T.to_csv('BoW-vocab1.csv',index=False)

#
# # 4. split
# train_size = X_train_BoW.shape[0]
# test_size = X_test_BoW.shape[0]
#
# ## random hold out
# ts = test_size/train_size
# X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_train_new,Y_train, test_size=ts)


# # ## k fold
# kf = KFold(n_splits=round(train_size/test_size)) # #test / #train
# for train, test in kf.split(X_train_new):
#     X_train_s, X_test_s, y_train_s, y_test_s = X_train_new[train], X_train_new[test], Y_train[train],Y_train[test]

# 5. modelling

## base model: 0R
# clf = DummyClassifier(strategy='most_frequent')
# basemodel = clf.fit(X_train_raw, Y_train)
# print("base model score: ", basemodel.score(X_train_raw, Y_train))

## other models


### tree


### logistic regression

# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import cross_val_score
# import numpy as np
# from sklearn.model_selection import GridSearchCV
# from sklearn.model_selection import RandomizedSearchCV
#
# hyper = {
#     'solver': ['newton-cg', 'lbfgs', 'sag', 'saga'],
#     'penalty': ['l1', 'l2', 'none', 'elasticnet'],
#     'C': [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000],
#     'max_iter': [100, 200, 300],
#     'multi_class': ['auto', 'ovr', 'multinomial']
# }
#
# # clf = GridSearchCV(LogisticRegression(),hyper, scoring='accuracy', cv=5)
# clf = RandomizedSearchCV(LogisticRegression(),hyper, scoring='accuracy', cv=5, n_iter=50)
# clf.fit(X_train_s, y_train_s)
# print(clf.cv_results_)
# print(pd.DataFrame(clf.cv_results_)[['solver','penalty', 'C', 'max_iter', 'multi_class']])

# logi_model = LogisticRegression(solver='sag', multi_class='multinomial', C=0.8, max_iter=200).fit(X_train_s, y_train_s)
# logi_model.predict(X_test_s)
# log_acc = np.mean(cross_val_score(logi_model,X_test_s,X_test_s,y_test_s,cv=5))
# print(f"logistic model score: { logi_model.score(X_test_s,y_test_s)}")

### svm
from sklearn.svm import SVC
svm_model = SVC(kernel="linear", C=0.1).fit(X_train_s, y_train_s)
svm_model.score(X_test_s,y_test_s)
print(f"svm model score: ", svm_model.score(X_test_s,y_test_s))
svm_hyper = {
    'degree': [3, 5, 10, 15],
    'gamma': [1,0.1,0.01,0.001],
    'C': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000],
    'kernel': ['linear', 'poly', 'rbf', 'sigmod'],
    'max_iter': [-1, 100, 500, 1000],
    'decision_function_shape': ['ovo', 'ovr']
}

search_svm = GridSearchCV(SVC(), svm_hyper, scoring='accuracy', cv=5)


#
#
# ### random forest
# from sklearn.ensemble import RandomForestClassifier

# rf_hyper = {
#     'n_estimators': [90, 100, 115 , 130],
#     'criterion': ['gini', 'entropy'],
#     'max_depth': range(2,20,1),
#     'min_sample_leaf': range(1,10,1),
#     'min_samples_split': range(2,10,1),
#     'max_features': ['auto', 'log2']
# }

# rf_model = RandomForestClassifier(criterion='entropy', max_depth=12,max_features='log2', min_samples_leaf=5, min_samples_split=5, n_estimators=90, random_state=6)
# rf_model.fit(X_train_s,y_train_s)
# print("rf model score: ", rf_model.score(X_test_s, y_test_s))
#
#
# ### stacking
# from sklearn.ensemble import StackingClassifier
#
# stacking_hyper = {
#
# }


# estimators = [('rf', rf_model),('svr', svm_model)]
#
# stacking_model = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(max_iter=200))
# stacking_model.fit(X_train_s, y_train_s)
#
# print("stacking model score: ", stacking_model.score(X_test_s, y_test_s))
#
#
#
#
# # evaluation
