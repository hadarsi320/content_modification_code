from multiprocessing import Lock

# Top Document Refinement Methods
VANILLA = 'vanilla'
ACCELERATION = 'acceleration'
PAST_TOP = 'past_top'
HIGHEST_RATED_INFERIORS = 'highest_rated_inferiors'
PAST_TARGETS = 'past_targets'
EVERYTHING = 'everything'

# Document Replacement Validation Methods
NAIVE = 'naive'
PROBABILITIES = 'probabilities'
PREDICTION = 'prediction'

# Lock for multiprocessing
lock = Lock()

# Machine Files
embedding_model_file = '/lv_local/home/hadarsi/work_files/word2vec_model/word2vec_model'
base_index = '/lv_local/home/hadarsi/work_files/clueweb_index/'
swig_path = '/lv_local/home/hadarsi/indri-5.6/swig/obj/java/'
indri_path = '/lv_local/home/hadarsi/indri/'

#   Project Files
project_path = '/lv_local/home/hadarsi/pycharm_projects/content_modification_code/'
# Data Files
queries_file = project_path + 'data/queries_seo_exp.xml'
stopwords_file = project_path + 'data/stopwords_list'
raifer_trec_file = project_path + 'data/trec_file_original_sorted.txt'
raifer_trectext_file = project_path + 'data/documents.trectext'
goren_positions_file = project_path + 'data/paper_data/documents.positions'
goren_trectext_file = project_path + 'data/paper_data/documents.trectext'
# Scripts
indri_utils_path = project_path + 'scripts/seo_indri_utils.jar'
