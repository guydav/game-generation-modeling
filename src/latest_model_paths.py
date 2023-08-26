import os


_FILE_DIR = os.path.dirname(__file__)


LATEST_REAL_GAMES_PATH = os.path.abspath(os.path.join(_FILE_DIR, '../dsl/interactive-beta.pddl'))

LATEST_FITNESS_FEATURES = os.path.abspath(os.path.join(_FILE_DIR,  '../data/fitness_features_1024_regrowths.csv.gz'))
LATEST_AST_N_GRAM_MODEL_PATH = os.path.abspath(os.path.join(_FILE_DIR, '../models/ast_5_ngram_model_2023_08_25.pkl'))
LATEST_FITNESS_FEATURIZER_PATH = os.path.abspath(os.path.join(_FILE_DIR, '../models/fitness_featurizer_2023_08_23.pkl.gz'))
LATEST_FITNESS_FUNCTION_DATE_ID = 'full_features_no_in_data_all_2023_08_21'

LATEST_FITNESS_FEATURES_SPECIFIC_OBJECTS_NGRAM = os.path.abspath(os.path.join(_FILE_DIR,  '../data/fitness_features_1024_regrowths_specific_objects_ngram.csv.gz'))
LATEST_SPECIFIC_OBJECTS_AST_N_GRAM_MODEL_PATH = os.path.abspath(os.path.join(_FILE_DIR, '../models/ast_specific_objects_5_ngram_model_2023_06_27.pkl'))
LATEST_SPECIFIC_OBJECTS_FITNESS_FEATURIZER_PATH = os.path.abspath(os.path.join(_FILE_DIR, '../models/fitness_featurizer_specific_objects_ngram_2023_07_21.pkl.gz'))
LATEST_SPECIFIC_OBJECTS_FITNESS_FUNCTION_DATE_ID = 'full_features_specific_objects_ngram_2023_07_21'
