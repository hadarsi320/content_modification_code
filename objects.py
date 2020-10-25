
class RankedLists:
    def __init__(self, raw=False, **kwargs):
        if 'trec_dir' in kwargs:
            trec_dir = kwargs.pop('trec_dir')
        elif 'trec_file' in kwargs:
            trec_file = kwargs.pop('trec_file')
        else:
            raise ValueError('RankedLists Object must receive either a trec_dir or a trec_file argument')
