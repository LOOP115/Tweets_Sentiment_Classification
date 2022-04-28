import numpy as np
from sklearn.model_selection import KFold


kf = KFold(n_splits=6099/21802) # #test / #train