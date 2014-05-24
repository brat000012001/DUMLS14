print (__doc__)

import argparse
import sys, os
import traceback
import logging
from datetime import date

import numpy as np
from numpy import histogram
from scipy.sparse import vstack, hstack
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
from sklearn.preprocessing import Normalizer, normalize, StandardScaler
from sklearn.random_projection import SparseRandomProjection,GaussianRandomProjection
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.kernel_approximation import (RBFSampler,Nystroem)
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, ClassifierMixin


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

class StandardScalerSVR:

	def __init__(self, **kwargs):
		self.estimator = Pipeline([
			#('feature_map', Nystroem(gamma=.2, random_state=1)),
			('svr', SVR())
		])

		self.estimator.set_params(**kwargs)

	def fit(self, X, y):
		return self.estimator.fit(X, y)

	def predict(self, X):
		return self.estimator.predict(X)

	def __repr_(self):
		return self.__str__()

	def __str__(self):
		return '{}'.format(repr(self.estimator))			

class KernelizedSGD:

	def __init__(self, **kwargs):
		self.estimator = Pipeline([
			('feature_map', Nystroem(gamma=.2, random_state=1)),
			('sgd', SGDClassifier())
		])

		self.estimator.set_params(**kwargs)

	def fit(self, X, y):
		return self.estimator.fit(X, y)

	def predict(self, X):
		return self.estimator.predict(X)

	def __repr_(self):
		return self.__str__()

	def __str__(self):
		return '{}'.format(repr(self.estimator))			


class MyOneVsRestClassifier(OneVsRestClassifier):

	def fit(self, X, y=None):
		r = OneVsRestClassifier.fit(self, X, y)
		logging.debug('Classes => %s' % (repr(self.classes_)))
		return r

	def predict(self, X):
		return OneVsRestClassifier.predict(self, X)		

#'''
class EnsembleClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, classifiers=None):
        self.classifiers_ = classifiers

    def fit(self, X, y):
        for classifier in self.classifiers:
            classifier.fit(X, y)

    def predict(self, X):
        self.predictions_ = list()
        for classifier in self.classifiers:
            self.predictions_.append(classifier.predict(X))
        #return np.mean(self.predictions_, axis=0)
#'''

def majority_vote(estimators, data, labels, metric=accuracy_score):

	all_predictions = []

	for estimator in estimators:
		prediction = estimator.predict(data)
		all_predictions.append(prediction)

	'''
	logging.debug('Individual classifiers reported the following scores:')
	logging.debug('\tScore1: %.6f,\tScore2: %.6f,\tScore3: %.6f' % (accuracy_score(labels, lbl_1),accuracy_score(labels, lbl_2),accuracy_score(labels, lbl_3)))
	'''

	votes_ = [[None,0,0,0,0,0] for i in range(len(labels))]

	for pred in all_predictions:
		for i in range(len(pred)):
			vote = votes_[i]
			vote[int(pred[i])] += 1	

	majority_vote = []		
	for vote in votes_:
		max_ = 0
		class_ = 1
		for i in range(len(vote)-1):
			if vote[i+1] > max_:
				class_ = i+1
				max_ = vote[i+1]				
		majority_vote.append(class_)	

	one_score = metric(labels, majority_vote)
	logging.debug('Best score: %.6f' % (one_score))

	return majority_vote

if __name__ == '__main__':
    

    try:
        parser = argparse.ArgumentParser(description='baseline for predicting labels')

        parser.add_argument('-d',
                            default='.',
                            help='Directory with datasets in SVMLight format')

        parser.add_argument('-id', type=int,
                            default=1,
                            choices=[1,2,3],
                            help='Dataset id')

        if len(sys.argv) == 1:
            parser.print_help()
            sys.exit(0)

        args = parser.parse_args()

	print os.path.basename(__file__)
	logging.basicConfig(filename='%s.log' % (os.path.basename(__file__)),format='%(asctime)s %(message)s',level=logging.DEBUG)
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
	logging.info(repr(args))



        from sklearn.datasets import dump_svmlight_file, load_svmlight_file

	#features = [None, 253659, 200, 100]
	features = [None, 200, 200, 100]
	#components = [None, 120, 200, 100]

	# load validation and test datasets using specified dataset ID

	if args.id == 1:

		fname_trn = os.path.join(args.d, "dt%d.%s.merged.svm" % (args.id, "trn"))
        	fname_vld = os.path.join(args.d, "dt%d.%s.merged.svm" % (args.id, "vld"))
        	fname_tst = os.path.join(args.d, "dt%d.%s.merged.svm" % (args.id, "tst"))
        
        	for fn in (fname_trn, fname_vld, fname_tst):
	            	if not os.path.isfile(fn):
	                	print("Missing dataset file: %s " % (fn,))
	                	sys.exit(1)

        	fname_vld_lbl = os.path.join(args.d, "dt%d.%s.lbl" % (args.id, "vld"))
        	fname_tst_lbl = os.path.join(args.d, "dt%d.%s.lbl" % (args.id, "tst"))

        	fname_vld_pred = os.path.join(args.d, "dt%d.%s.pred" % (args.id, "vld"))
        	fname_tst_pred = os.path.join(args.d, "dt%d.%s.pred" % (args.id, "tst"))

        	### reading labels
		logging.info("Loading training, validation and testing datasets...")
        	data_trn, lbl_trn = load_svmlight_file(fname_trn, n_features=features[args.id], zero_based=True)
        	data_vld, lbl_vld = load_svmlight_file(fname_vld, n_features=features[args.id], zero_based=True)
        	data_tst, lbl_tst = load_svmlight_file(fname_tst, n_features=features[args.id], zero_based=True)

		#'''
		data_trn = data_trn.toarray()
		data_vld = data_vld.toarray()
		data_tst = data_tst.toarray()
		#'''
		pca = PCA(n_components=100)
		pca.fit(data_trn)
		data_trn = pca.transform(data_trn)
		data_vld = pca.transform(data_vld)
		data_tst = pca.transform(data_tst)

		# LinearSVC to classify dataset #1
		cls1 = OneVsRestClassifier(LinearSVC(C=1.0))
		logging.debug('Fitting dataset #1 with %s...' % (cls1))
		cls1.fit(data_trn,lbl_trn)
		
		#  SVC to classify dataset #2
		fname_trn2 = os.path.join(args.d, "dt%d.%s.merged.svm" % (2, "trn"))
        	data_trn2, lbl_trn2 = load_svmlight_file(fname_trn2, n_features=features[2], zero_based=True)
		data_trn2 = data_trn2.toarray()
		pca = PCA(n_components=100)
		pca.fit(data_trn2)
		data_trn2 = pca.transform(data_trn2)

		cls2 = OneVsRestClassifier(SGDClassifier())
		logging.debug('Fitting dataset #2 with %s...' % (cls2))
		cls2.fit(data_trn2,lbl_trn2)

		# SVC to classify dataset #3
		fname_trn3 = os.path.join(args.d, "dt%d.%s.merged.svm" % (3, "trn"))
        	data_trn3, lbl_trn3 = load_svmlight_file(fname_trn3, n_features=features[3], zero_based=True)
		data_trn3 = data_trn3.toarray()
		cls3 = OneVsOneClassifier(SGDClassifier())
		logging.debug('Fitting dataset #3 with %s...' % (cls3))
		cls3.fit(data_trn3,lbl_trn3)

		
		cls4 = SVC(C=1.0)
		cls4.fit(data_trn, lbl_trn)

		cls5 = KNeighborsClassifier()
		cls5.fit(data_trn, lbl_trn)

		cls6 = OneVsOneClassifier(SGDClassifier())
		cls6.fit(data_trn2, lbl_trn2)

		cls7 = OneVsRestClassifier(SGDClassifier())
		cls7.fit(data_trn2,lbl_trn2)

		cls8 = SVC(C=10.0)
		cls8.fit(data_trn, lbl_trn)

		cls9 = OneVsOneClassifier(SGDClassifier())
		cls9.fit(data_trn, lbl_trn)

		cls10 = ExplicitFeatureMapApproximation(svm__C=1.0)
		cls10.fit(data_trn, lbl_trn)

		cls11 = ExplicitFeatureMapApproximation(svm__C=1.0)
		cls11.fit(data_trn2, lbl_trn2)

		cls12 = ExplicitFeatureMapApproximation(svm__C=1.0)
		cls12.fit(data_trn3, lbl_trn3)

		cls13 = KNeighborsClassifier()
		cls13.fit(data_trn2, lbl_trn2)

		cls14 = KNeighborsClassifier()
		cls14.fit(data_trn3, lbl_trn3)

		vld_lbl_ = majority_vote((cls1, cls2, cls3, cls4, cls5, cls6, cls7, cls8, cls9, cls10, cls11, cls12, cls13, cls14), data_vld, lbl_vld)
		tst_lbl_ = majority_vote((cls1, cls2, cls3, cls4, cls5, cls6, cls7, cls8, cls9, cls10, cls11, cls12, cls13, cls14), data_tst, lbl_tst)


		format_obj = '%d'

	        np.savetxt(fname_vld_pred, vld_lbl_, delimiter='\n', fmt=format_obj)
		np.savetxt(fname_tst_pred, tst_lbl_, delimiter='\n', fmt=format_obj)
		np.savetxt(fname_vld_lbl, lbl_vld, delimiter='\n', fmt=format_obj)
		np.savetxt(fname_tst_lbl, lbl_tst, delimiter='\n', fmt=format_obj)



	elif args.id == 3:

		fname_trn = os.path.join(args.d, "dt%d.%s.merged.svm" % (args.id, "trn"))
        	fname_vld = os.path.join(args.d, "dt%d.%s.merged.svm" % (args.id, "vld"))
        	fname_tst = os.path.join(args.d, "dt%d.%s.merged.svm" % (args.id, "tst"))
        
        	for fn in (fname_trn, fname_vld, fname_tst):
	            	if not os.path.isfile(fn):
	                	print("Missing dataset file: %s " % (fn,))
	                	sys.exit(1)

        	fname_vld_lbl = os.path.join(args.d, "dt%d.%s.lbl" % (args.id, "vld"))
        	fname_tst_lbl = os.path.join(args.d, "dt%d.%s.lbl" % (args.id, "tst"))

        	fname_vld_pred = os.path.join(args.d, "dt%d.%s.pred" % (args.id, "vld"))
        	fname_tst_pred = os.path.join(args.d, "dt%d.%s.pred" % (args.id, "tst"))

        	### reading labels
		logging.info("Loading training, validation and testing datasets...")
        	data_trn, lbl_trn = load_svmlight_file(fname_trn, n_features=features[args.id], zero_based=True)
        	data_vld, lbl_vld = load_svmlight_file(fname_vld, n_features=features[args.id], zero_based=True)
        	data_tst, lbl_tst = load_svmlight_file(fname_tst, n_features=features[args.id], zero_based=True)

		#'''
		data_trn = data_trn.toarray()
		data_vld = data_vld.toarray()
		data_tst = data_tst.toarray()
		#'''

		'''
		pca = PCA(n_components=100)
		pca.fit(data_trn)
		data_trn = pca.transform(data_trn)
		data_vld = pca.transform(data_vld)
		data_tst = pca.transform(data_tst)
		'''
		# LinearSVC to classify dataset #1
		cls1 = OneVsRestClassifier(LinearSVC(C=1.0))
		logging.debug('Fitting dataset #1 with %s...' % (cls1))
		cls1.fit(data_trn,lbl_trn)
		
		#  SVC to classify dataset #2
		fname_trn2 = os.path.join(args.d, "dt%d.%s.merged.svm" % (1, "trn"))
        	data_trn2, lbl_trn2 = load_svmlight_file(fname_trn2, n_features=features[1], zero_based=True)
		data_trn2 = data_trn2.toarray()

		pca = PCA(n_components=100)
		pca.fit(data_trn2)
		data_trn2 = pca.transform(data_trn2)

		cls2 = OneVsRestClassifier(SGDClassifier())
		logging.debug('Fitting dataset #2 with %s...' % (cls2))
		cls2.fit(data_trn2,lbl_trn2)

		# SVC to classify dataset #3
		fname_trn3 = os.path.join(args.d, "dt%d.%s.merged.svm" % (2, "trn"))
        	data_trn3, lbl_trn3 = load_svmlight_file(fname_trn3, n_features=features[2], zero_based=True)
		data_trn3 = data_trn3.toarray()

		pca = PCA(n_components=100)
		pca.fit(data_trn3)
		data_trn3 = pca.transform(data_trn3)

		cls3 = OneVsOneClassifier(SGDClassifier())
		logging.debug('Fitting dataset #3 with %s...' % (cls3))
		cls3.fit(data_trn3,lbl_trn3)

		
		cls4 = SVC(C=1.0)
		cls4.fit(data_trn, lbl_trn)

		cls5 = KNeighborsClassifier()
		cls5.fit(data_trn, lbl_trn)

		cls6 = OneVsOneClassifier(SGDClassifier())
		cls6.fit(data_trn2, lbl_trn2)

		cls7 = OneVsRestClassifier(SGDClassifier())
		cls7.fit(data_trn2,lbl_trn2)

		cls8 = SVC(C=10.0)
		cls8.fit(data_trn, lbl_trn)

		cls9 = OneVsOneClassifier(SGDClassifier())
		cls9.fit(data_trn, lbl_trn)

		cls10 = ExplicitFeatureMapApproximation(svm__C=1.0)
		cls10.fit(data_trn, lbl_trn)

		cls11 = ExplicitFeatureMapApproximation(svm__C=1.0)
		cls11.fit(data_trn2, lbl_trn2)

		cls12 = ExplicitFeatureMapApproximation(svm__C=1.0)
		cls12.fit(data_trn3, lbl_trn3)

		cls13 = KNeighborsClassifier()
		cls13.fit(data_trn2, lbl_trn2)

		cls14 = KNeighborsClassifier()
		cls14.fit(data_trn3, lbl_trn3)

		vld_lbl_ = majority_vote((cls1, cls2, cls3, cls4, cls5, cls6, cls7, cls8, cls9, cls10, cls11, cls12, cls13, cls14), data_vld, lbl_vld, mean_squared_error)
		tst_lbl_ = majority_vote((cls1, cls2, cls3, cls4, cls5, cls6, cls7, cls8, cls9, cls10, cls11, cls12, cls13, cls14), data_tst, lbl_tst, mean_squared_error)


		format_obj = '%.6f'

	        np.savetxt(fname_vld_pred, vld_lbl_, delimiter='\n', fmt=format_obj)
		np.savetxt(fname_tst_pred, tst_lbl_, delimiter='\n', fmt=format_obj)
		np.savetxt(fname_vld_lbl, lbl_vld, delimiter='\n', fmt=format_obj)
		np.savetxt(fname_tst_lbl, lbl_tst, delimiter='\n', fmt=format_obj)



    except Exception, exc:
        print('Exception was raised in %s of %s: %s \n %s ' % (__name__, __file__, str(exc), ''.join(traceback.format_exc())))






