import strax
import numpy as np
import os
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
    # new plugins must be listed here in order to be registered
    plugins_to_register = 'DAQReader Records Peaks PeakBasics PeakClassification ' \
                          'NCompeting Events EventHermBasics EventHermPositions'.split(' ')

    def __init__(self, *args, **kwargs):
        experiment = kwargs.pop('experiment', 'htpc')

        if 'config' not in kwargs:
            kwargs['config'] = {'experiment': experiment}
        elif 'experiment' not in kwargs['config']:
            kwargs['config']['experiment'] = experiment
        if 'storage' not in kwargs:
            kwargs['storage'] = os.path.join(storage_base_dir, experiment)

        # add configs that must be (/you wanto to be) different from xebra dual phase tpc here
        # n_channels and drift_length/drift_vel is set in utils.py - relies on run_id starting with '1'
        configs = {                  # type       plugin       xebra_val
                'hit_threshold': 7,  #  int   records,peaks       30
                'top_pmts': [1]      # list   p_basics,p_pos    list(range(1,8))
                }

        for config_name, config_value in configs.items():
            if config_name not in kwargs['config']:
                kwargs['config'][c_name] = c_value
        
        if 'register' not in kwargs and 'register_all' not in kwargs:
            register = []
            # mostly stolen from strax's context.py
            for plugin in dir(plugins):
                plugin = getattr(plugins, plugin)
                if type(plugin) != type(type):
                    continue
                if (
                        issubclass(plugin, strax.Plugin) and
                        plugin.__name__ in self.plugins_to_register):

                    register.append(plugin)
            kwargs['register'] = register

        super().__init__(*args, **kwargs)
              






