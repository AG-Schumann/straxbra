import strax
import numpy as np
import os
import pandas as pd
from . import utils

export, __all__ = strax.exporter()


def process_runlist(run_id):
    if isinstance(run_id, str):
        _validate_id(run_id)
    else:
        for run in run_id:
            _validate_id(run)

    if isinstance(run_id, str) and len(run_id) > 7:
        runs_list = []
        for i in range(0, len(run_id), 5):
            runs_list.append(f'{int(run_id[i:i+5], 10):05d}')
        return runs_list
    return run_id


def _validate_id(run_id):
    if len(run_id) % 5 != 0:
         raise ValueError(
                    'All Run_IDs have 5 digits (zero-padded).'
                    'Expected len of multi-run call to be divisible by 5.')


def update(d, u):
    """Deep-updates dicts. Dicts as values are updated instead of being replaced."""
    for k, v in u.items():
        if isinstance(v, dict):
            d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

@export
class XebraContext(strax.Context):
    """Context for dual-phase xebra tpc

    This is the context to use for standard dual-phase
    operation of the xebra TPC. It is also the base from
    which other contexts for different TPCs should inherit.
    """

    std_dtypes = ('raw_records', 'records', 'peaks', 'events', 'event_info')

    # Settings here override defaults but can themselves be
    # overriden by argument config passed to __init__
    config = {
        'experiment': 'xebra'
    }
    storage = '/data/storage/strax/cached/xebra'

    def __init__(self, *args, **kwargs):
        kwargs['config'] = update(self.config, kwargs.pop('config', {}))
        self.runs_db = RunsDBInterface(self['config']['experiment'])

        if 'storage' not in kwargs:
            kwargs['storage'] = self.storage
        if 'register' not in kwargs and 'register_all' not in kwargs:
            # Importing now allows us to access context in plugin options
            # Not really sure if this is a good idea
            from . import plugins
            kwargs['register_all'] = plugins

        super().__init__(*args, **kwargs)

    def get_array(self, run_id, *args, **kwargs) -> np.ndarray:
        run_id = process_runlist(run_id)
        return super().get_array(run_id, *args, **kwargs)

    def get_df(self, run_id, *args, **kwargs) -> pd.DataFrame:
        run_id = process_runlist(run_id)
        return super().get_df(run_id, *args, **kwargs)




@export
class HtpcContext(XebraContext):
    """Context for hermetic tpc

    Main differences from XebraContext:
      - Two PMTs, one top and one bottom
      - Different drift_length
      - Updated strax tuning parameters (config)
    """

    std_dtypes = ('raw_records', 'records', 'peaks', 'events')

    # add configs that must be (/you wanto to be) different from xebra dual phase tpc here
    config = {                              # type       plugin       xebra_val
            'experiment' : 'xebra_hermetic_tpc',
            'hit_threshold': 7,              #  int   records,peaks       30
            'top_pmts': [1],                 # list   p_basics,p_pos    list(range(1,8))
            'min_reconstruction_area': 1e10, #  this makes sure no pos-reconst. is attempted
            # always use that model so the hash won't change if another exp changes the model:
            'nn_model': 'fake_htpc_model_not_actually_used_but_must_exist.h5'
            }
    storage = '/data/storage/strax/cached/htpc'

    def __init__(self, *args, **kwargs):
        utils.experiment = 'xebra_hermetic_tpc'
        utils.n_pmts = 2
        utils.drift_length = 7.8  # in cm

        super().__init__(*args, **kwargs)


@export
class SinglePhaseContext(XebraContext):
    """Context for single phase operation of a xebra tpc

    Use this context to tune strax for single-phase operation.
    Inherits basically everything from XebraContext.
    """

    config = {
        'experiment': 'xebra_singlephase'
    }
    storage = '/data/storage/strax/cached/xebra_singlephase'

    def __init__(self, *args, **kwargs):
        utils.experiment = 'xebra_singlephase'
        super().__init__(*args, **kwargs)
