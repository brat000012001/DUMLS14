import argparse
import sys, os

import numpy as np
from numpy import histogram
#from scipy.stats import itemfreq
from sklearn.externals import joblib
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import ParameterGrid
from sklearn.svm import LinearSVC, SVC, SVR
#from sklearn.manifold import Isomap
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.feature_selection import SelectKBest, SelectPercentile, chi2
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.decomposition import SparsePCA, PCA
## peter LDA requires dense matrix
#from sklearn.lda import LDA
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import grid_search
from sklearn.feature_selection import RFE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OutputCodeClassifier
from sklearn.ensemble import ExtraTreesClassifier

class MyPipeline:
	def __init__(self, **kwargs):

		k = 10
		self.dict = dict()
		for key,value in kwargs.iteritems():
			if key != 'k':
				self.dict[key] = value
			else:
				k = value

		#self.estimator = RFE(LinearSVC(),step=1000,estimator_params=self.dict)
		
		self.estimator = Pipeline(
		[
			#("feature_selection", SelectKBest(chi2,k=k)),
			#("feature_selection", TruncatedSVD(n_components=100,algorithm="arpack",random_state=0,tol=0)),
			#("outputcode", OutputCodeClassifier(LinearSVC(**self.dict),code_size=4,random_state=0))
			#("dimensionality_reduction",TruncatedSVD(n_components=100)),
			#("feature_selection", LinearSVC(penalty="l1",dual=False,C=1)),
			("svc", SVC(**self.dict))
			#("sgd", SGDClassifier(**self.dict))
		])
		

	def fit(self, x, y):
		return self.estimator.fit(x,y)

	def predict(self,x):
		return self.estimator.predict(x)


def train(args, grid_obj, cls_obj, metric_obj, data_trn, lbl_trn, data_vld, lbl_vld):

        best_param = None
        best_score = None
        best_svc = None

        for one_param in ParameterGrid(grid_obj):

		try:
			cls = cls_obj(**one_param)
			cls.fit(data_trn, lbl_trn)
			one_score = metric_obj(lbl_vld, cls.predict(data_vld))

			print ("param=%s, score=%.6f" % (repr(one_param),one_score))
            
			if ( best_score is None or (args.id < 3 and best_score < one_score) or (args.id == 3 and best_score > one_score) ):
				best_param = one_param
				best_score = one_score
				best_svc = cls
		except KeyboardInterrupt:
			raise
		except Exception as e:
			print "Exception due to invalid parameter combination: {}".format(e)
	return (best_param, best_score, best_svc)


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

        parser.add_argument('-scale', 
                            action='store_true',
                            default=False,
                            help='Classifier id')

        parser.add_argument('-reduce', 
                            action='store_true',
                            default=False,
                            help='Whether to reduce dimensionality of the input data.')

        parser.add_argument('-plotsvd',
                            action='store_true',		
                            default=False,
                            help='Visualizes SVD-reprojected dataset #1')

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

	args.id = 1 # override whatever user's input since we only be experimenting with dataset #1 here...
        n_features = 253659

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

	# peter print unique labels. Gradient-descent classifier does not seem to support class_weight parameter...?? 
	#print np.unique(lbl_trn)
	#print np.ones(lbl_trn.shape[0], dtype=np.float64, order='C')
	#print np.searchsorted(lbl_trn, '0.')
	#print np.searchsorted(lbl_trn, '1.')
	#print np.searchsorted(lbl_trn, '2.')
	#print np.searchsorted(lbl_trn, '3.')
	#print np.searchsorted(lbl_trn, '4.')
	#print np.searchsorted(lbl_trn, '5.')

	# Check if the data needs to be re-scaled
	if args.scale:
		print "Rescaling the input data, please wait..."
		scaler = StandardScaler()
		data_trn = scaler.fit_transform(data_trn)
		data_vld = scaler.fit_transform(data_vld)
		data_tst = scaler.fit_transform(data_tst)


	# Plot the principal components using SVD
	if args.plotsvd:
		try:
			import pylab as pl


			# 2 eigenvectors for the sake of 2D visualization
			svd = TruncatedSVD(n_components=2)
			X_r2 = svd.fit(data_trn, lbl_trn).transform(data_trn)
			pl.figure()
			for c, i, target_name in zip("rgb", [1, 2, 3, 4, 5], set(lbl_trn)):
				pl.scatter(X_r2[lbl_trn == i, 0], X_r2[lbl_trn == i, 1], c=c, label=target_name)
			pl.legend()
			pl.title('SVD of Dataset#1 dataset')
			pl.show()

			pl.figure()

			at_hist, xedges, yedges = np.histogram2d(data_trn, lbl_trn, bins=500)
			extent = [yedges[0], yedges[-1], xedges[0], xedges[-1]]

			fig = mpl.pylab.figure()
			at_plot = fig.add_subplot(111)
			at_plot.imshow(at_hist, extent=extent, origin='lower', aspect='auto')

			pl.show()

		except Exception as e:
			print 'Check if pylab is installed. %s' % (str(e))

        
        #metrics = (None, accuracy_score, accuracy_score, accuracy_score, accuracy_score, accuracy_score, accuracy_score, accuracy_score, accuracy_score, accuracy_score, mean_squared_error)
        #str_formats = (None, "%d", "%d", "%d","%d","%d","%d","%d","%d","%d","%d","%d","%d","%d","%d","%d","%d","%.6f")
        #LinearSVC(penalty='l2', loss='l2', dual=True, tol=0.0001, C=1.0,

        #grid_obj=grids[args.cid]
        #cls_obj=classifiers[args.cid]
        #metric_obj=metrics[args.cid]

        best_param = None
        best_score = None
        best_svc = None
	best_n_components = None

	data_trn_ = data_trn
	data_vld_ = data_vld
	data_tst_ = data_tst

	str_formats = ["%d","%.6f"]

	grid_obj = [{'class_weight':['auto'],'C':[100,1000,10000,100000,1000000,10000000],'cache_size':[1024],'gamma':[0.0001,0.001,0.01,0.1], 'tol':[0.001],'probability':[False],'kernel':['rbf']}] # ,'degree':[3,5,10,100]
	cls_obj = SVC
	metric_obj = accuracy_score
	format_obj = str_formats[0]

	if args.verbose:
		grid_obj[0]['verbose'] = [True]
        
	if args.reduce:
			
		n_components = 200
		
		svd = TruncatedSVD(n_components=n_components,algorithm="arpack",tol=0)
		# Perform dimensionality reduction
		svd.fit(data_trn)
		print "Reducing dimensionality of the training data to {} best vectors, please wait...".format(n_components)
		data_trn_ = svd.transform(data_trn)
		print "Reducing dimensionality of the validation data to {} best vectors, please wait...".format(n_components)
		data_vld_ = svd.transform(data_vld)
		print "Reducing dimensionality of the test data to {} best vectors, please wait...".format(n_components)
		data_tst_ = svd.transform(data_tst)
		
		pca = PCA(100)
		pca.fit(data_trn_)
		data_trn_ = pca.transform(data_trn_)
		data_vld_ = pca.transform(data_vld_)
		data_tst_ = pca.transform(data_tst_)
		

	print "\nClassifier:{}".format(repr(cls_obj))
	print "Metrics: {}".format(repr(metric_obj))        
	print "Grid:{}\n\n".format(repr(grid_obj))
	print "Training dataset: {}".format(fname_trn)
	print "Validation dataset: {}".format(fname_vld)
	print "Testing dataset: {}".format(fname_tst)

	print "Commencing the training/validation phase..."
	best_param, best_score, best_svc = train(args, grid_obj, cls_obj, metric_obj, data_trn_, lbl_trn, data_vld_, lbl_vld)

		  
        pred_vld = best_svc.predict(data_vld_)
        pred_tst = best_svc.predict(data_tst_)

	print "\n\nBest configuration: {}".format(repr(best_param))        
        print ("Best score for vld: %.6f" % (metric_obj(lbl_vld, pred_vld),))
        print ("Best score for tst: %.6f" % (metric_obj(lbl_tst, pred_tst),))


        np.savetxt(fname_vld_pred, pred_vld, delimiter='\n', fmt=format_obj)
        np.savetxt(fname_tst_pred, pred_tst, delimiter='\n', fmt=format_obj)
        np.savetxt(fname_vld_lbl, lbl_vld, delimiter='\n', fmt=format_obj)
        np.savetxt(fname_tst_lbl, lbl_tst, delimiter='\n', fmt=format_obj)

    except Exception, exc:
        import traceback
        print('Exception was raised in %s of %s: %s \n %s ' % (__name__, __file__, str(exc), ''.join(traceback.format_exc())))






