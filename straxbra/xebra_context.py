import strax
import numpy as np
import os
import pandas as pd
from . import utils

export, __all__ = strax.exporter()
from . import plugins


def process_runlist(run_id):
    if isinstance(run_id, str) and len(run_id) > 7:
        if len(run_id) % 5 != 0:
            raise ValueError(
                        'All Run_IDs have 5 digits (zero-padded).'
                        'Expected len of multi-run call to be divisible by 5.')
        runs_list = []
        for i in range(0, len(run_id), 5):
            runs_list.append(f'{int(run_id[i:i+5], 10):05d}')
        return runs_list
    return run_id


def update(d, u):
    """Deep-updates dicts. Dicts as values are updated instead of being replaced."""
    for k, v in u.items():
        if isinstance(v, dict):
            d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


storage_base_dir = '/data/storage/strax/cached/'


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
            kwargs['storage'] = os.path.join(storage_base_dir, experiment)
        if 'register' not in kwargs and 'register_all' not in kwargs:
            kwargs['register_all'] = plugins
        super().__init__(*args, **kwargs)

    def get_array(self, run_id, *args, **kwargs) -> np.ndarray:
        run_id = process_runlist(run_id)
        return super().get_array(run_id, *args, **kwargs)

    def get_df(self, run_id, *args, **kwargs) -> pd.DataFrame:
        run_id = process_runlist(run_id)
        return super().get_df(run_id, *args, **kwargs)


@export
class HtpcContext(strax.Context):

    std_dtypes = ('raw_records', 'records', 'peaks', 'events')

    def __init__(self, *args, **kwargs):
        utils.experiment = 'htpc'
        utils.n_pmts = 2
        utils.drift_length = 7.8  # in cm

        # add configs that must be (/you wanto to be) different from xebra dual phase tpc here
        configs = {                              # type       plugin       xebra_val
                'hit_threshold': 7,              #  int   records,peaks       30
                'top_pmts': [1],                 # list   p_basics,p_pos    list(range(1,8))
                'min_reconstruction_area': 1e10, #  this makes sure no pos-reconst. is attempted
                # always use that model so the hash won't change if another exp changes the model:
                'nn_model': 'fake_htpc_model_not_actually_used_but_must_exist.h5'
                }

        experiment = kwargs.pop('experiment', 'htpc')
        standards = {'storage': os.path.join(storage_base_dir, experiment),
                     'register_all': plugins,
                     'config': update({'experiment': experiment}, configs)}

        kwargs = update(standards, kwargs)
        super().__init__(*args, **kwargs)
