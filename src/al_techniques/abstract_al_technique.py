

class AbstractALTechnique:

    def __init__(self, model):
        self.model = model

    def get_score(self, sample, target_type):
        pass
