import os
import sys
import re
import numpy as np
from random import shuffle

import pandas as pd
import argparse
from config_training import config 
import utils2 as utils
from evaluator import s_to_p

parser = argparse.ArgumentParser(description='PyTorch DataBowl3 Detector')

parser.add_argument('--filename', default=None, type=str, metavar='PATH',
                    help='directory to load the original filenames (default: filenames.npy)')
parser.add_argument('--save-dir', default=None, type=str, metavar='PATH',
                    help='directory to save the split filenames (default: filenames.npy)')
parser.add_argument('--pred-dir', default=None, type=str, metavar='PATH',
                    help='directory to save the split filenames (default: filenames.npy)')

def main():
    #Split the dataset into K folders
    args = parser.parse_args()
    
    filename = args.filename
    filenames = np.load(args.filename)
    save_dir = args.save_dir
    bbox_dir = args.pred_dir
    k = 1000
    PREP = config['preprocess_result_path']
    name_map = pd.read_csv('luna_shorter.csv', dtype=str)
    # print name_map

    submission = []

    for name in filenames:
        pbb = np.load(os.path.join(bbox_dir, "%s_pbb.npy" % (name)))
        pbb = pbb[pbb[:, 0].argsort()][::-1][:k]
        spacing = np.load(os.path.join(PREP, name + '_spacing.npy'))
        ebox_origin = np.load(os.path.join(PREP, name + '_ebox_origin.npy'))
        origin = np.load(os.path.join(PREP, name + '_origin.npy'))
        #print spacing

        for p in pbb:
            ebox_coord = p[[1, 2, 3]]
            whole_img_coord = ebox_coord + ebox_origin
            worldCoord = utils.voxelToWorldCoord(whole_img_coord, origin, spacing)
            # print name_map[name_map['short'] == name]['long']
            submission.append([name_map[name_map['short'] == name]['long'].as_matrix()[0], worldCoord[2], worldCoord[1], worldCoord[0], p[0]])

    submission = pd.DataFrame(submission, columns = ["seriesuid", "coordX", "coordY", "coordZ",
                                                     "probability"])
    scores = np.array(submission['probability'])
    probs = s_to_p(scores)
    submission['probability'] = probs

    print "Saving submission to", save_dir
    submission.to_csv(save_dir, sep=',', index=False, header=False)

if __name__ == '__main__':
    main()
