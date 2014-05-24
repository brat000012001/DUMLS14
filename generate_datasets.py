print (__doc__)

import argparse
import sys, os
import traceback
import logging
from datetime import date

import numpy as np
from numpy import histogram
from scipy.sparse import vstack, hstack
from sklearn.decomposition import SparsePCA, PCA, KernelPCA,RandomizedPCA,TruncatedSVD


if __name__ == '__main__':
    

    try:
        parser = argparse.ArgumentParser(description='generates concatenated datasets')

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
	logging.info(repr(args))



        from sklearn.datasets import dump_svmlight_file, load_svmlight_file


	features = [None, 253659, 200, 100]

	# load validation and test datasets using specified dataset ID

	fname_trn = os.path.join(args.d, "dt%d.%s.svm" % (args.id, "trn"))
        fname_vld = os.path.join(args.d, "dt%d.%s.svm" % (args.id, "vld"))
        fname_tst = os.path.join(args.d, "dt%d.%s.svm" % (args.id, "tst"))
        
        for fn in (fname_trn, fname_vld, fname_tst):
            if not os.path.isfile(fn):
                print("Missing dataset file: %s " % (fn,))
                sys.exit(1)

        fname_vld_lbl = os.path.join(args.d, "dt%d.%s.lbl" % (args.id, "vld"))
        fname_tst_lbl = os.path.join(args.d, "dt%d.%s.lbl" % (args.id, "tst"))

        fname_vld_pred = os.path.join(args.d, "dt%d.%s.pred" % (args.id, "vld"))
        fname_tst_pred = os.path.join(args.d, "dt%d.%s.pred" % (args.id, "tst"))

        ### reading labels
	logging.info("Loading master dataset {}...".format(fname_trn))
        data_trn, lbl_trn = load_svmlight_file(fname_trn, n_features=features[args.id], zero_based=True)
        data_vld, lbl_vld = load_svmlight_file(fname_vld, n_features=features[args.id], zero_based=True)
        data_tst, lbl_tst = load_svmlight_file(fname_tst, n_features=features[args.id], zero_based=True)


	if args.id == 1:
		svd = TruncatedSVD(n_components=200)
		svd.fit(data_trn)
		data_trn = svd.transform(data_trn)
		data_vld = svd.transform(data_vld)
		data_tst = svd.transform(data_tst)

		fname = os.path.join(args.d, "dt%d.%s.svm" % (2, "trn"))
		logging.info("Loading '{}' training dataset...".format(fname))
        	trn, lbl = load_svmlight_file(fname, n_features=features[2], zero_based=True)

		trn = trn.toarray()

		logging.info("Concatenating training datasets {} and {}...".format(data_trn.shape,trn.shape))
		data_trn = np.vstack((data_trn,trn))
		lbl_trn = np.hstack([lbl_trn, lbl])
		logging.info('Concatenated datasets: {}'.format(data_trn.shape))
		logging.info('Concatenated labels: {}'.format(lbl_trn.shape))

	elif args.id == 2:
		svd = TruncatedSVD(n_components=100)
		svd.fit(data_trn)
		data_trn = svd.transform(data_trn)
		data_vld = svd.transform(data_vld)
		data_tst = svd.transform(data_tst)

		fname = os.path.join(args.d, "dt%d.%s.svm" % (1, "trn"))
		logging.info("Loading '{}' training dataset...".format(fname))
        	trn, lbl = load_svmlight_file(fname, n_features=features[1], zero_based=True)

		svd = TruncatedSVD(n_components=100)
		svd.fit(trn)
		trn = svd.transform(trn)

		logging.info("Concatenating training datasets {} and {}...".format(data_trn.shape,trn.shape))
		data_trn = np.vstack((data_trn,trn))
		lbl_trn = np.hstack([lbl_trn, lbl])
		logging.info('Concatenated datasets: {}'.format(data_trn.shape))
		logging.info('Concatenated labels: {}'.format(lbl_trn.shape))
	
	elif args.id == 3:

		fname = os.path.join(args.d, "dt%d.%s.svm" % (1, "trn"))
		logging.info("Loading '{}' training dataset...".format(fname))
        	trn, lbl = load_svmlight_file(fname, n_features=features[1], zero_based=True)

		svd = TruncatedSVD(n_components=100)
		svd.fit(trn)
		trn = svd.transform(trn)

		pca = PCA(n_components=100)
		pca.fit(trn)
		trn = pca.transform(trn)

		logging.info("Concatenating training datasets {} and {}...".format(data_trn.shape,trn.shape))
		data_trn = vstack((data_trn,trn))
		lbl_trn = np.hstack([lbl_trn, lbl])
		logging.info('Concatenated datasets: {}'.format(data_trn.shape))
		logging.info('Concatenated labels: {}'.format(lbl_trn.shape))


		fname = os.path.join(args.d, "dt%d.%s.svm" % (2, "trn"))
		logging.info("Loading '{}' training dataset...".format(fname))
        	trn, lbl = load_svmlight_file(fname, n_features=features[2], zero_based=True)

		trn = trn.toarray()
		pca = PCA(n_components=100)
		pca.fit(trn)
		trn = pca.transform(trn)

		logging.info("Concatenating training datasets {} and {}...".format(data_trn.shape,trn.shape))
		data_trn = vstack((data_trn,trn))
		lbl_trn = np.hstack([lbl_trn, lbl])
		logging.info('Concatenated datasets: {}'.format(data_trn.shape))
		logging.info('Concatenated labels: {}'.format(lbl_trn.shape))


	fname_trn_output = os.path.join(args.d, "dt%d.%s.merged.svm" % (args.id, "trn"))
	vname_trn_output = os.path.join(args.d, "dt%d.%s.merged.svm" % (args.id, "vld"))
	tname_trn_output = os.path.join(args.d, "dt%d.%s.merged.svm" % (args.id, "tst"))
	dump_svmlight_file(data_trn,lbl_trn, fname_trn_output)
	dump_svmlight_file(data_vld,lbl_vld, vname_trn_output)
	dump_svmlight_file(data_tst,lbl_tst, tname_trn_output)

    except Exception, exc:
        print('Exception was raised in %s of %s: %s \n %s ' % (__name__, __file__, str(exc), ''.join(traceback.format_exc())))






