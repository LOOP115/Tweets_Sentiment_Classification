import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import load_digits
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split


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


# 2. vectorization (transformation)

# bag of words
# countvectorizer
BoW_vectorizer = CountVectorizer(analyzer='word', ngram_range=(2,2))
X_train_BoW = BoW_vectorizer.fit_transform(X_train_raw)
X_test_BoW = BoW_vectorizer.transform(X_test_raw)
print("Train feature space size (using bow):",X_train_BoW.shape)
print("Test feature space size (using bow):",X_test_BoW.shape)

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



# 3. feature selection
X_train_new = SelectKBest(chi2,k=5000).fit_transform(X_train_BoW,Y_train)
print(X_train_new[0].toarray())
print(type(X_train_new[0]))

# # save bow
# output_dict = BoW_vectorizer.vocabulary_
# output_pd = pd.DataFrame(list(output_dict.items()),columns = ['word','count'])
# output_pd.T.to_csv('BoW-vocab1.csv',index=False)

# 4. train

# split
train_size = X_train_BoW.shape[0]
test_size = X_test_BoW.shape[0]

## random hold out
ts = test_size/train_size
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_train_new,Y_train, test_size=ts)


# # ## k fold
kf = KFold(n_splits=round(train_size/test_size)) # #test / #train
for train, test in kf.split(X_train_new):
    X_train_s, X_test_s, y_train_s, y_test_s = X_train_new[train], X_train_new[test], Y_train[train],Y_train[test]

# model selection