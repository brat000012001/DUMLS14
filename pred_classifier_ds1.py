print (__doc__)

import argparse
import sys, os
import traceback
import logging
from datetime import date

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
from sklearn.preprocessing import Normalizer
from sklearn.random_projection import SparseRandomProjection,GaussianRandomProjection
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from sklearn.kernel_approximation import (RBFSampler,Nystroem)

def my_kernel(x, y):
	"""
	We create a custom kernel:

			(2  0)
	k(x, y) =    x  (    ) y.T
			(0  1)
	"""
	M = np.array([[2, 0], [0, 1.0]])
	return np.dot(np.dot(x, M), y.T)


class L1LinearSVC(LinearSVC):

	def fit(self, X, Y):
		self.transformer_ = LinearSVC(penalty="l1",dual=False,tol=1e-3)
		X = self.transformer_.fit_transform(X, Y)
		return LinearSVC.fit(self, X, Y)

	def predict(self, X):
		X = self.transformer_.transform(X)
		return LinearSVC.predict(self, X)

class SvdSVC:

	def __init__(self, **kwargs):

		self.classifier_ = Pipeline([
			('feature_selection', LinearSVC()),
			('svc', SVC())
		])
		self.classifier_.set_params(**kwargs)

		#SVC.__init__(self, **kwargs)
		#self.transformer_ = LinearSVC(C=1.0,penalty="l1",dual=False,tol=1e-3)

	def fit(self, X, y):
		return self.classifier_.fit(X, y)
		#X = self.transformer_.fit_transform(X, y)
		#return SVC.fit(self, X, y)

	def predict(self, X):
		return self.classifier_.predict(X)
		#X = self.transformer_.transform(X)
		#return SVC.predict(self, X)

	def __repr__(self):
		return self.__str__()

	def __str__(self):
		return '{}+{}'.format(repr(self.classifier_))

class ExplicitFeatureMapApproximation:

	def __init__(self,**kwargs):
		#self.feature_map_ = RBFSampler(gamma=.2, random_state=1)
		self.feature_map_ =  Nystroem(gamma=.2, random_state=1)
		self.classifier_ = Pipeline([
			('feature_map', self.feature_map_),
			('svm', LinearSVC())])

		self.classifier_.set_params(**kwargs)

	def fit(self, X, y):
		return self.classifier_.fit(X, y)

	def predict(self, X):
		return self.classifier_.predict(X)

	def __repr__(self):
		return self.__str__()

	def __str__(self):
		return '{}'.format(repr(self.classifier_))


class TruncatedSVDReducer:

	def __init__(self, **kwargs):
		self.svd_ = TruncatedSVD(**kwargs)
		self.normalizer_ = normalize

	def fit(self, X, y = None):
		print "Reducing dimensionality of the training data to {} best vectors, please wait...".format(self.svd_.n_components)
		return self.svd_.fit(X, y)

	def transform(self, X):
		print "Reducing dimensionality of the validation data to {} best vectors, please wait...".format(self.svd_.n_components)
		X_ =  self.svd_.transform(X)
		# Independently normalize each feature
		#return normalize(X_,axis=0,copy=False)

		# Independently normalize each feature
		return normalize(X_,axis=1,copy=False)

	def __repr__(self):
		return self.__str__()

	def __str__(self):
		return '{}+{}'.format(str(self.svd_),repr(self.normalizer_))

def train(args, reducer_obj, grid_obj, cls_obj, metric_obj, data_trn, lbl_trn, data_vld, lbl_vld):

        best_param = None
        best_score = None
        best_svc = None
	best_reducer = None

	if reducer_obj == None:

	        for one_param in ParameterGrid(grid_obj):

			try:
				cls = cls_obj(**one_param)
				cls.fit(data_trn, lbl_trn)
				one_score = metric_obj(lbl_vld, cls.predict(data_vld))

				logging.info("param=%s, score=%.6f" % (repr(one_param),one_score))
            
				if ( best_score is None or (args.id < 3 and best_score < one_score) or (args.id == 3 and best_score > one_score) ):
					best_param = one_param
					best_score = one_score
					best_svc = cls
			except KeyboardInterrupt:
				raise
			except Exception, exc:
				print('Exception was raised in %s of %s: %s \n %s ' % (__name__, __file__, str(exc), ''.join(traceback.format_exc())))

	else:
		for reducer in reducer_obj:

			# Perform dimensionality reduction
			reducer.fit(data_trn)
			data_trn_ = reducer.transform(data_trn)
			data_vld_ = reducer.transform(data_vld)

		        for one_param in ParameterGrid(grid_obj):

				try:
					cls = cls_obj(**one_param)
					cls.fit(data_trn_, lbl_trn)
					one_score = metric_obj(lbl_vld, cls.predict(data_vld_))

					logging.info("dimensionality reduction: %s, param=%s, score=%.6f" % (repr(reducer),repr(one_param),one_score))
            
					if ( best_score is None or (args.id < 3 and best_score < one_score) or (args.id == 3 and best_score > one_score) ):
						best_param = one_param
						best_score = one_score
						best_svc = cls
						best_reducer = reducer
				except KeyboardInterrupt:
					raise
				except Exception, exc:
					print('Exception was raised in %s of %s: %s \n %s ' % (__name__, __file__, str(exc), ''.join(traceback.format_exc())))


	return (best_param, best_score, best_svc, best_reducer)

def gridsearch_n_train(args, grid_obj, cls_obj, metric_obj, data_trn, lbl_trn, data_vld, lbl_vld):

        best_param = None
        best_score = None
        best_svc = None

	try:
		cls = GridSearchCV(estimator=cls_obj(),param_grid=grid_obj,iid=False,n_jobs=3)
		cls.fit(data_trn,lbl_trn)
		one_score = cls.best_score_
		best_score = metric_obj(lbl_vld, cls.best_estimator_.predict(data_vld))
		best_param = cls.best_params_	
		best_svc = cls.best_estimator_
		logging.info("param=%s, score (cv)=%.6f, score=%.6f" % (repr(cls.best_params_),one_score, best_score))
	except KeyboardInterrupt:
		raise
	except Exception, exc:
	        print('Exception was raised in %s of %s: %s \n %s ' % (__name__, __file__, str(exc), ''.join(traceback.format_exc())))
	return (best_param, best_score, best_svc)


if __name__ == '__main__':
    
    logging.basicConfig(filename='classification.log',format='%(asctime)s %(message)s',level=logging.DEBUG)
    # define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    # set a format which is simpler for console use
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    # tell the handler to use this format
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger('').addHandler(console)
    logging.info("---------------------------------------------------------------------")
    logging.info('{}'.format('--->>> CLASSIFICATION <<<---'))
    logging.info("---------------------------------------------------------------------")


    try:
        parser = argparse.ArgumentParser(description='baseline for predicting labels')

        parser.add_argument('-d',
                            default='.',
                            help='Directory with datasets in SVMLight format')

        parser.add_argument('-id', type=int,
                            default=1,
                            choices=[1,2,3],
                            help='Dataset id')

        parser.add_argument('-reduce', 
                            type=int,
                            default=None,
                            choices=[1,2,3,4],
                            help='Whether to reduce dimensionality of the input data.')

        parser.add_argument('-full', 
                            action='store_true',
                            default=False,
                            help='Whether to use original training dataset or only a small sub-set of the original dataset.')

        parser.add_argument('-verbose', 
                            action='store_true',
                            default=False,
                            help='Verbose output.')

        if len(sys.argv) == 1:
            parser.print_help()
            sys.exit(0)

        args = parser.parse_args()
	logging.info(repr(args))

	n_features = 100
        if args.id == 1:
		n_features = 253659
	elif args.id == 2:
		n_features = 200	

	# For datasets #2 and #3 use the entire dataset (not a subset)
	if args.id != 1: 
		args.full = True

	subset = ""
	if args.full == False:
		subset = "_1500"
        fname_trn = os.path.join(args.d, "dt%d%s.%s.svm" % (args.id, subset, "trn"))
        fname_vld = os.path.join(args.d, "dt%d%s.%s.svm" % (args.id, subset, "vld"))
        fname_tst = os.path.join(args.d, "dt%d%s.%s.svm" % (args.id, subset, "tst"))

        fname_vld_lbl = os.path.join(args.d, "dt%d%s.%s.lbl" % (args.id, subset, "vld"))
        fname_tst_lbl = os.path.join(args.d, "dt%d%s.%s.lbl" % (args.id, subset, "tst"))

        fname_vld_pred = os.path.join(args.d, "dt%d%s.%s.pred" % (args.id, subset, "vld"))
        fname_tst_pred = os.path.join(args.d, "dt%d%s.%s.pred" % (args.id, subset, "tst"))
        
        for fn in (fname_trn, fname_vld, fname_tst):
            if not os.path.isfile(fn):
                print("Missing dataset file: %s " % (fn,))
                sys.exit(1)
        
        ### reading labels
        from sklearn.datasets import dump_svmlight_file, load_svmlight_file
        data_trn, lbl_trn = load_svmlight_file(fname_trn, n_features=n_features, zero_based=True)
        data_vld, lbl_vld = load_svmlight_file(fname_vld, n_features=n_features, zero_based=True)
        data_tst, lbl_tst = load_svmlight_file(fname_tst, n_features=n_features, zero_based=True)

	# Print weights for each class
	uniq_keys = np.unique(lbl_trn)
	bins = uniq_keys.searchsorted(lbl_trn)
	# [1 2 3 4 5] -> [584 584 15 147 170]
	logging.info('Training data: {} -> {}'.format(repr(uniq_keys), repr(np.bincount(bins))))
	# Print the same statistics but for validation data set
	uniq_keys = np.unique(lbl_vld)
	bins = uniq_keys.searchsorted(lbl_vld)
	logging.info('Validation data: {} -> {}'.format(repr(uniq_keys), repr(np.bincount(bins))))

	# Note that the probability of samples belonging to class '3' is 6 times less likely than other classes.
	class_weights = {1:1,2:1,3:7.0,4:3.5,5:3.5}
	class_weights1 = {1:1,2:1,3:8.0,4:3.5,5:3.5}
	class_weights2 = {1:1,2:1,3:16,4:3.5,5:3.5}
	class_weights3 = {1:1,2:1,3:32.,4:3.5,5:3.5}

	# SVC: 0.48820 with cosine_similarity as kernel, C = 100        
	#grid1_obj = [{'C':[1.0,1e1,1e2,1e3,1e4,1e5],'cache_size':[1024],'gamma':[0.1], 'tol':[0.001],'probability':[False],'kernel':[cosine_similarity]}]

	# LinearSVC: with dimensionality reduction can get 0.43
	#grid1_obj = [{'C':[1.0,1e1,1e2,1e3,1e4,1e5],'class_weight':[class_weights]}]

	# SvdSVC, 0.50-0.54: uses TruncatedSVD  to reduce the dimensionality and custom class weights set propertionally to the ratio of the most frequent class over class frequency 
	grid1_obj = [{'C':[1.0,1e1,1e2],'cache_size':[1024],'gamma':[0.1,1.0],'tol':[0.001],'probability':[False],'kernel':[cosine_similarity,'rbf'],'class_weight':[class_weights]}]

	# SvdSVC (using pipeline)
	grid1_obj = [{'svc__C':[1.0,1e1,1e2],'svc__cache_size':[1024],'svc__gamma':[0.1,1.0],'svc__tol':[0.001],'svc__probability':[False],'svc__kernel':[cosine_similarity,'rbf'],
		'svc__class_weight':[class_weights],'feature_selection__C':[0.1,1.0,10.0],'feature_selection__penalty':["l1"],'feature_selection__dual':[False],'feature_selection__tol':[1e-3]}]

	# SGDClassifier
	#grid1_obj = [{'loss':['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'],'penalty':['l1','l2'],'alpha':[0.0001,0.001,0.01,0.1,0.00001],'class_weight':[class_weights]}]

	# Explicit Kernel Approximation
	#grid1_obj = [{'feature_map__n_components':[250,500,1000,2000,3000],'svm__C':[1.0,1e1,1e2],'svm__class_weight':[class_weights]}]
	##grid1_obj = [{'feature_map__n_components':[100,500,1000,2000,3000],'svm__C':[1.0,1e1,1e2],'svm__gamma':[0.1,1.0],'svm__kernel':[cosine_similarity,'rbf','poly'],'svm__class_weight':[class_weights]}]

	
	# LogisticRegression
	#grid1_obj = [{'base_estimator':[LogisticRegression(C=1e1),LogisticRegression(C=1e2),LogisticRegression(C=1e3),LogisticRegression(C=1e4)]}]
	grid2_obj = [{'n_neighbors':[3,5,10],'algorithm':['auto', 'ball_tree', 'kd_tree', 'brute'],'p':[1,2,3]}]
	#grid3_obj = [{'C':[1e-1,1,1e1,1e2,1e3,1e4,1e5,1e6,1e7],'kernel':['rbf'],'gamma':[1e-1,0,10,100],'cache_size':[1024],'nu':[0.8,1.0]}]
	# DecisionTreeClassifier(max_depth=5),DecisionTreeClassifier(max_depth=10),DecisionTreeClassifier(max_depth=50)
	#grid3_obj = [{'base_estimator':[SelectKBest()], 'n_estimators':[300, 600, 1000, 10000], 'learning_rate':[0.5, 1.5, 3], 'algorithm':["SAMME"]}]
	#grid3_obj = [{'radius':[1.0,10],'weights':['distance'],'algorithm':['auto','ball_tree','kd_tree'],'p':[2],'outlier_label':[1,2,3,4,5]}]
	grid3_obj = [{'C':[1e1,1e2,1e3,1e4,1e5],'kernel':['rbf']}]
	str_formats = [None,"%d","%d","%.6f"]

	grids = [None, grid1_obj, grid2_obj, grid3_obj]
	classifiers = [None, SvdSVC, ExplicitFeatureMapApproximation, SGDClassifier, LinearSVC, KNeighborsClassifier, SVR]
        metrics = (None, accuracy_score, accuracy_score, mean_squared_error)

	# 320,300, 280 is too much!
	reducer1_obj = [TruncatedSVDReducer(n_components=252)] # TruncatedSVDReducer(n_components=100),TruncatedSVDReducer(n_components=50)
	reducer2_obj = [TruncatedSVDReducer(n_components=150)]

	dimensionality_reducers = [None, reducer1_obj, reducer2_obj, None]

        grid_obj=grids[args.id]
        cls_obj=classifiers[args.id]
        metric_obj=metrics[args.id]
	format_obj = str_formats[args.id]
	reducer_obj = dimensionality_reducers[args.id]

	if args.verbose:
		grid_obj[0]['verbose'] = [True]

        best_param = None
        best_score = None
        best_svc = None
	best_n_components = None

	data_trn_ = data_trn
	data_vld_ = data_vld
	data_tst_ = data_tst

	logging.info( "Classifier:{}".format(str(cls_obj)))
	logging.info( "Metrics: {}".format(str(metric_obj)))        
	logging.info( "Grid:{}".format(str(grid_obj)))
	logging.info( "Reducer:{}".format(str(reducer_obj)))
	logging.info( "Training dataset: {}".format(fname_trn))
	logging.info( "Validation dataset: {}".format(fname_vld))
	logging.info( "Testing dataset: {}".format(fname_tst))

	print "Commencing the training/validation phase..."
	#best_param, best_score, best_svc = train(args, grid_obj, cls_obj, metric_obj, data_trn_, lbl_trn, data_vld_, lbl_vld)
	best_param, best_score, best_svc, best_reducer = train(args, reducer_obj, grid_obj, cls_obj, metric_obj, data_trn_, lbl_trn, data_vld_, lbl_vld)


	if best_reducer != None: 
	        data_vld_ = best_reducer.transform(data_vld_)
		data_tst_ = best_reducer.transform(data_tst_)


        pred_vld = best_svc.predict(data_vld_)
        pred_tst = best_svc.predict(data_tst_)

	logging.info("\n\nBest Classifier: {}".format(repr(best_svc)))       
	logging.info("\n\nBest Reducer: {}".format(repr(best_reducer)))       
	logging.info("\n\nBest configuration: {}".format(repr(best_param)))       
        logging.info("Best score for vld: %.6f" % (metric_obj(lbl_vld, pred_vld)))
        logging.info("Best score for tst: %.6f" % (metric_obj(lbl_tst, pred_tst)))


        np.savetxt(fname_vld_pred, pred_vld, delimiter='\n', fmt=format_obj)
        np.savetxt(fname_tst_pred, pred_tst, delimiter='\n', fmt=format_obj)
        np.savetxt(fname_vld_lbl, lbl_vld, delimiter='\n', fmt=format_obj)
        np.savetxt(fname_tst_lbl, lbl_tst, delimiter='\n', fmt=format_obj)

    except Exception, exc:
        print('Exception was raised in %s of %s: %s \n %s ' % (__name__, __file__, str(exc), ''.join(traceback.format_exc())))






