import os


_FILE_DIR = os.path.dirname(__file__)

LATEST_AST_N_GRAM_MODEL_PATH = os.path.abspath(os.path.join(_FILE_DIR, '../models/ast_7_ngram_model_2023_04_16.pkl'))
LATEST_FITNESS_FEATURIZER_PATH = os.path.abspath(os.path.join(_FILE_DIR, '../models/fitness_featurizer_2023_04_07.pkl.gz'))
LATEST_FITNESS_FUNCTION_DATE_ID = 'full_features_2023_04_07'
LATEST_FITNESS_FEATURES =  os.path.abspath(os.path.join(_FILE_DIR,  '../data/fitness_features.csv.gz'))
