from models._stat_model_base import StatHeadModel


class Model(StatHeadModel):
    def __init__(self, configs):
        super().__init__(configs, encoder_name='Prophet')
