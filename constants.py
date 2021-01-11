from multiprocessing import Lock

# Top Document Refinement Methods
VANILLA = 'vanilla'
ACCELERATION = 'acceleration'
PAST_TOP = 'past_top'
HIGHEST_RATED_INFERIORS = 'highest_rated_inferiors'
PAST_TARGETS = 'past_targets'
EVERYTHING = 'everything'

# Document Replacement Validation Methods
PROBABILITIES = 'probabilities'
OPTIMISTIC = 'optimistic'
PREDICTION = 'prediction'

# Lock for multiprocessing
lock = Lock()
