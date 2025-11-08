USER = 'user_id'
ITEM = 'item_id'
AUTHOR = 'author_id'
EMBEDDING = 'embedding'
TIME_INDEX = 'time_index'

TARGET = "rank"
INTERACTIONS_NUM_FEATURES = ['timespent']
INTERACTIONS_CAT_FEATURES = ['place', 'platform', 'agent', ]

USERS_NUM_FEATURES = ['age']
USERS_CAT_FEATURES = ['gender', 'geo']

ITEMS_NUM_FEATURES = ['duration']
ITEMS_CAT_FEATURES = []

INTERACTIONS_MAP = {
    'like': 2.0, 'share': 5.0, 'bookmark': 4.0, 'click_on_author': 3.0, 'open_comments': 1.5,
    'dislike': -5.0
}

DATA_DIR = "VK-LSVD"

RANDOM_STATE = 42