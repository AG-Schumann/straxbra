import strax
import numpy as np
import tqdm
import pandas as pd

export, __all__ = strax.exporter()
from . import plugins

def process_runlist(run_id):
    if isinstance(run_id, str) and len(run_id) > 7:
        runs_list = []
        for i in range(0, len(run_id), 4):
            runs_list.append(f'{int(run_id[i:i+4], 16):05d}')
        return runs_list
    return run_id

@export
class XebraContext(strax.Context):

    std_dtypes = ('raw_records', 'records', 'peaks', 'events', 'event_info')

    def __init__(self, *args, **kwargs):
        experiment = kwargs.pop('experiment', 'xebra')
        if 'config' not in kwargs:
            kwargs['config'] = {'experiment' : experiment}
        elif 'experiment' not in kwargs['config']:
            kwargs['config']['experiment'] = experiment
        if 'storage' not in kwargs:
            kwargs['storage'] = f'/data/storage/strax/cached/{experiment}/'
        if 'register' not in kwargs and 'register_all' not in kwargs:
            kwargs['register_all'] = plugins
        super().__init__(*args, **kwargs)

    def get_array(self, run_id, *args, **kwargs) -> np.ndarray:
        run_id = process_runlist(run_id)
        return super().get_array(run_id, *args, **kwargs)

    def get_df(self, run_id, *args, **kwargs) -> pd.DataFrame:
        run_id = process_runlist(run_id)
        return super().get_df(run_id, *args, **kwargs)
