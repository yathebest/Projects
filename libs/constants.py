TARGET = "rank"
USER = 'user_id'
ITEM = 'item_id'
AUTHOR = 'author_id'
EMBEDDING = 'embedding'
TIME_INDEX = 'time_index'

CATEGORICAL = 'categorical'
NUMERICAL = 'numerical'
TEMPORAL = 'temporal'

INTERACTION_FEATURES = {
    CATEGORICAL: ['place', 'platform', 'agent'],
    NUMERICAL: ['timespent', 'timeratio'],
    TEMPORAL: [TIME_INDEX],
}

USER_FEATURES = {
    CATEGORICAL: ['gender', 'geo', 'age'],
    NUMERICAL: [],
    TEMPORAL: []
}

ITEM_FEATURES = {
    CATEGORICAL: [],
    NUMERICAL: ['duration'],
    TEMPORAL: [],
}

TARGET_MAP = {
    'like': 2.0, 'share': 5.0, 'bookmark': 4.0, 'click_on_author': 3.0, 'open_comments': 1.5,
    'dislike': -5.0
}

MAX_PER_USER = 100
TOP_PER_ITEM = 100

DATA_DIR = "VK-LSVD"
MODELS_DIR = "checkpoints"
CACHE_DIR = "cache"

EPS = 1e-8
RANDOM_STATE = 42