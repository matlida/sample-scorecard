import os
print(1)

class ModelAbc(object):
    global abspath
    abspath = os.path(os.path.dirname('__file__'))

    def __init__(self, threshold = 0.5,feature_renaming = False):
        self.threshold = threshold
        self.feature_renaming = feature_renaming
        log_path = os.path.join(abspath,'log')
        if not os.path.exists(log_path):
            os.makedirs(log_path)