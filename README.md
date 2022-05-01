# Sentiment Classification of Tweets

* [Specification](spec/ass2_spec.pdf)

## Progress Check List
> **Data Preprocessing**
* Vectorization
  * Bag of Words
  * TFIDF
  * Doc2Vec
  * Word2Vec
* Data Cleaning (冲榜必备!)
  ~~V Remove URL tags~~ 
  * Remove extra whitespaces
  * Convert accented characters to ASCII characters
  * Expand contractions
  ~~V Remove special characters~~
  ~~V Lowercase all texts~~
  * Convert number words to numeric form
  * Remove numbers
  * Remove stopwords
  * Lemmatization
* https://towardsdatascience.com/nlp-text-preprocessing-a-practical-guide-and-template-d80874676e79
* https://gist.github.com/jiahao87/d57a2535c2ed7315390920ea9296d79f

> **Feature Engineering**
* SelectKBest
  * Chi-Square (chi2)
  * F-Test (f_classif)
  * Mutual Information (mutual_info_classif)
* Variance Threshold
* https://scikit-learn.org/stable/modules/classes.html#module-sklearn.feature_selection

> **~~Data Splitting~~**
* Maximize Validation Accuracy
* Choice of Test Size: 6099 / 21802
* (Random) Hold-out
* K-fold Cross-Validation
* https://scikit-learn.org/stable/modules/cross_validation.html?highlight=cross_validation#cross-validation-and-model-selection

> **Machine Learning Models**
~~* Baseline Model: 0-R~~
* Naive Bayes
* KNN
* Decision Tree
* ~~Logistic Regression~~
* ~~SVM~~
* ~~Random Forest~~
* ~~Stacking~~
* Hyperparameter Tuning: Grid Search

> **Evaluation**
* Micro/Macro/Weighted Averaging
* ROC Curve
* Time Complexity

> **Error Analysis**
* Confusion Matrix

> **Report**
* Introduction
* Method
* Results
* Discussion / Critical Analysis
* Conclusion
* Bibliography  _(Include Rosenthal, 2017)_

