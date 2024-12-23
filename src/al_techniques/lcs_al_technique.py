from src.al_techniques.abstract_al_technique import AbstractALTechnique


class LCSALTechnique(AbstractALTechnique):

    def get_score(self, sample, target_type):
        return self.model(sample.x_dict, sample.edge_index_dict)[0][target_type].max(axis=1).values.cpu().numpy() * -1 #TODO to remove -1 made for sociologi
