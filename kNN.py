import numpy as np
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import KFold, cross_val_score


#KF = KFold(n=150, n_folds=num_fold, shuffle=True)

def main():
    iris = datasets.load_iris()
    k=input("Number of neighbors:")
    n=input("Number of folds:")
    print(oss_error(k,n,iris.data, iris.target))
    

def oss_error(k,n, X,Y):
    """ Does n-fold cross validation using knn and returns the mean score.
        Inputs: 
            k (int) - the number of nearest neighbors used to vote
            n (int) - the number of folds to use
        Returns:
            The mean of the knn classifier scores for held-out validation sets.
    """
    knn = KNeighborsClassifier(n_neighbors = k)
    scores = cross_val_score(knn, X, Y, cv=n)
    return scores.mean()

if __name__ == "__main__":
    main()
  