print(X_train_need_to_clean)
# url: r"\b*https?:\S*"
# hash tag and at asperand
#
# remove url, # and @
X_train_need_to_clean.replace("\b*https?:\S*", '', regex=True, inplace=True)
X_train_need_to_clean.replace("\b*@\S*", '', regex=True, inplace=True)
X_train_need_to_clean.replace("\b*#\S*", '', regex=True, inplace=True)

X_test_need_to_clean.replace("\b*https?:\S*", '', regex=True, inplace=True)
X_test_need_to_clean.replace("\b*@\S*", '', regex=True, inplace=True)
X_test_need_to_clean.replace("\b*#\S*", '', regex=True, inplace=True)

# lower case
X_train_need_to_clean[0] = X_train_need_to_clean[0].str.lower()
X_test_need_to_clean[0] = X_test_need_to_clean[0].str.lower()

k;''

# for i in range(X_train_need_to_clean.shape[0]):
#     X_train_need_to_clean.loc[i,0] =


def decontracted(phrase):
    # specific
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase


# ### naive bayes
# from sklearn.naive_bayes import GaussianNB
# gnb = GaussianNB()
# # gnb.fit(X_train_s, y_train_s.todense())
# # # y_pred_nb = gnb.predict(X_test_s, y_test_s)
# # gnb.score(X_test_s, y_test_s)