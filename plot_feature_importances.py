print(__doc__)

import os,sys
import argparse
import numpy as np
import traceback
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.decomposition import SparsePCA, PCA, KernelPCA,RandomizedPCA,TruncatedSVD


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

        parser.add_argument('-reduce', 
                            type=int,
                            default=None,
                            choices=[1,2,3,4],
                            help='Whether to reduce dimensionality of the input data.')


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
	if args.id != 1: 
		args.full = True

	subset = ""
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

	X, y = data_trn, lbl_trn

	# peter ExtraTreesClassifier cannot handle sparse matrices, so we need to reduce the dimensionality first
	svd = TruncatedSVD(n_components=100)
	X = svd.fit_transform(X)

	# Build a forest and compute the feature importances
	forest = ExtraTreesClassifier(n_estimators=250, random_state=0)

	forest.fit(X, y)
	importances = forest.feature_importances_
	std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
	indices = np.argsort(importances)[::-1]

	# Print the feature ranking
	print("Feature ranking:")

	for f in range(10):
		print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

	# Plot the feature importances of the forest
	import pylab as pl
	pl.figure()
	pl.title("Feature importances")
	pl.bar(range(10), importances[indices],color="r", yerr=std[indices], align="center")
	pl.xticks(range(10), indices)
	pl.xlim([-1, 10])
	pl.show()

    except Exception, exc:
        print('Exception was raised in %s of %s: %s \n %s ' % (__name__, __file__, str(exc), ''.join(traceback.format_exc())))
