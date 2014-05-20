print (__doc__)

import argparse
import sys, os
import traceback
from pprint import pprint

import numpy as np
from numpy import histogram
#from scipy.stats import itemfreq
from sklearn.externals import joblib
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import ParameterGrid,GridSearchCV
from sklearn.svm import LinearSVC, SVC, SVR, NuSVR
#from sklearn.manifold import Isomap
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.feature_selection import SelectKBest, SelectPercentile, chi2
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.decomposition import SparsePCA, PCA, KernelPCA,RandomizedPCA,TruncatedSVD
## peter LDA requires dense matrix
#from sklearn.lda import LDA
from sklearn.linear_model import SGDClassifier,LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.neighbors import KNeighborsClassifier,RadiusNeighborsClassifier
from sklearn.multiclass import OutputCodeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import Normalizer,normalize
from sklearn.random_projection import SparseRandomProjection,GaussianRandomProjection
from sklearn.metrics.pairwise import cosine_similarity

def classify(data_trn,lbl_trn,data_vld,lbl_vld,data_tst,lbl_tst):

	data_trn = normalize(data_trn,copy=False)
	data_vld = normalize(data_vld,copy=False)
	data_tst = normalize(data_tst,copy=False)

	# accuracy metric
	metric_obj = mean_squared_error
	'''
	Train our model to predict labels for the dataset #1
	'''
	parameters = {'svr__gamma': 1.5, 'svr__probability': False, 'svr__epsilon': 0.4, 'svr__C': 1, 'svr__kernel': 'rbf'}
	cls = Pipeline([
			#('feature_selection',LinearSVC()),
			('svr', SVR())
			])
	cls.set_params(**parameters)

	cls.fit(data_trn, lbl_trn)


        pred_vld = cls.predict(data_vld)
        pred_tst = cls.predict(data_tst)

        print ("Score for vld: %.6f" % (metric_obj(lbl_vld, pred_vld),))
        print ("Score for tst: %.6f" % (metric_obj(lbl_tst, pred_tst),))

	return pred_vld,pred_tst

if __name__ == '__main__':
    
    try:
        parser = argparse.ArgumentParser(description='baseline for predicting labels')

        parser.add_argument('-d',
                            default='.',
                            help='Directory with datasets in SVMLight format')

        parser.add_argument('-id', type=int,
                            default=3,
                            choices=[3],
                            help='Dataset id')

        if len(sys.argv) == 1:
            parser.print_help()
            sys.exit(0)

        args = parser.parse_args()

	n_features = 100
        if args.id == 1:
		n_features = 253659
	elif args.id == 2:
		n_features = 200	

	# For datasets #2 and #3 use the entire dataset (not a subset)

        fname_trn = os.path.join(args.d, "dt%d.%s.svm" % (args.id, "trn"))
        fname_vld = os.path.join(args.d, "dt%d.%s.svm" % (args.id, "vld"))
        fname_tst = os.path.join(args.d, "dt%d.%s.svm" % (args.id, "tst"))

        fname_vld_lbl = os.path.join(args.d, "dt%d.%s.lbl" % (args.id, "vld"))
        fname_tst_lbl = os.path.join(args.d, "dt%d.%s.lbl" % (args.id, "tst"))

        fname_vld_pred = os.path.join(args.d, "dt%d.%s.pred" % (args.id, "vld"))
        fname_tst_pred = os.path.join(args.d, "dt%d.%s.pred" % (args.id, "tst"))
        
        for fn in (fname_trn, fname_vld, fname_tst):
            if not os.path.isfile(fn):
                print("Missing dataset file: %s " % (fn,))
                sys.exit(1)
        
        ### reading labels
        from sklearn.datasets import dump_svmlight_file, load_svmlight_file
        data_trn, lbl_trn = load_svmlight_file(fname_trn, n_features=n_features, zero_based=True)
        data_vld, lbl_vld = load_svmlight_file(fname_vld, n_features=n_features, zero_based=True)
        data_tst, lbl_tst = load_svmlight_file(fname_tst, n_features=n_features, zero_based=True)

	format_obj = '%.6f'

	pred_vld,pred_tst = classify(data_trn,lbl_trn,data_vld,lbl_vld,data_tst,lbl_tst)

        np.savetxt(fname_vld_pred, pred_vld, delimiter='\n', fmt=format_obj)
        np.savetxt(fname_tst_pred, pred_tst, delimiter='\n', fmt=format_obj)
        np.savetxt(fname_vld_lbl, lbl_vld, delimiter='\n', fmt=format_obj)
        np.savetxt(fname_tst_lbl, lbl_tst, delimiter='\n', fmt=format_obj)

    except Exception, exc:
        print('Exception was raised in %s of %s: %s \n %s ' % (__name__, __file__, str(exc), ''.join(traceback.format_exc())))






