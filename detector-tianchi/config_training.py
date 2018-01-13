# For test2 data switch preprocessing
"""
config = {'preprocess_result_path':'/home/danielrk/lung/input/tianchi_prep/',
          'tianchi_data_path':'/home/danielrk/lung/input/tianchi_data_test2/',
          'tianchi_annos_path':['/home/danielrk/lung/input/htang6/csv/train/annotations.csv',
              '/home/danielrk/lung/input/htang6/csv/val/annotations.csv'],
          'tianchi_subsets_path':'/home/danielrk/lung/input/tianchi_data_test2/',
          'bbox_path':'../detector/results/05-31-17/bbox/train/'}
          """
# Preprocessing using preserved HU in dilated part of mask
config = {'preprocess_result_path':'/home/htang6/workspace/data/luna_1x1x1/',
          'tianchi_data_path':'/home/danielrk/lung/input/tianchi_combined/',
          'tianchi_annos_path':['/home/danielrk/lung/input/htang6/csv/train/annotations.csv',
              '/home/danielrk/lung/input/htang6/csv/val/annotations.csv'],
          'tianchi_subsets_path':'/home/danielrk/lung/input/htang6/',
          'bbox_path':'../detector/results/05-31-17/bbox/train/'}

# test1 + val + train tianchi
"""
config = {'preprocess_result_path':'/home/danielrk/lung/input/tianchi_prep/',
          'tianchi_data_path':'/home/danielrk/lung/input/tianchi_combined/',
          'tianchi_annos_path':['/home/danielrk/lung/input/htang6/csv/train/annotations.csv',
              '/home/danielrk/lung/input/htang6/csv/val/annotations.csv'],
          'tianchi_subsets_path':'/home/danielrk/lung/input/htang6/',
          'bbox_path':'../detector/results/05-31-17/bbox/train/'}
"""

luna_config = {'subsets_path':'/home/danielrk/lung/input/luna16/',
               'abbrev_path':'/home/danielrk/tc/nod/training/luna_shorter.csv'}
