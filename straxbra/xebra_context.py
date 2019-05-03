import strax
import numpy as np
import tqdm
import pandas as pd

export, __all__ = strax.exporter()


@export
class XebraContext(strax.Context):

    std_dtypes = ('raw_events', 'records', 'peaks', 'events', 'event_info')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_array(self, run_id, *args, **kwargs) -> np.ndarray:
        if isinstance(run_id, (str, int)):
            return super().get_array(run_id, *args, **kwargs)
        return np.concatenate([super().get_array(run,
                                                 *args,
                                                 **kwargs)
                               for run in tqdm.tqdm(run_id)])

    def get_df(self, run_id, *args, **kwargs) -> pd.DataFrame:
        if isinstance(run_id, (str, int)):
            return super().get_df(run_id, *args, **kwargs)
        return pd.DataFrame().append([super().get_df(run,
                                                     *args,
                                                     **kwargs)
                                            for run in tqdm.tqdm(run_id)],
                                     ignore_index=True)
