import os
import shutil

import numpy as np
from scipy.ndimage.interpolation import shift
from scipy.optimize import curve_fit

import numba

from . import utils
import strax

export, __all__ = strax.exporter()

# V/adc * (sec/sample) * (1/resistance) * (1/electron charge) * (amplification)
adc_to_e = (2.25/2**14) * (1e-8) * (1/50) * (1/1.602e-19) * (10)



@export
@strax.takes_config(
    strax.Option('input_dir', type=str, track=False,
                 default_by_run=utils.GetRawPath,
                 help='The directory with the data'),
    strax.Option('experiment', type=str, track=False, default='xebra',
                 help='Which experiment\'s data to load'),
    strax.Option('readout_threads', type=int, track=False,
                 default_by_run=utils.GetReadoutThreads,
                 help='How many readout threads were used'),
    strax.Option('safe_break', default=1000, track=False,
                 help='Time in ns between pulse starts indicating a safe break'),
    strax.Option('do_breaks', default=True, track=False,
                 help='Do the pulse breaking'),
    strax.Option('erase_reader', default=False, track=False,
                 help='Delete reader data after processing'),
    strax.Option('run_start', type=int, track=False,
                 default_by_run=utils.GetRunStart,
                 help='Start time of the run in ns'),
)
class DAQReader(strax.ParallelSourcePlugin):
    """
    Reads records in from disk. A nearly identical copy of straxen.DAQReader
    """
    provides = 'raw_records'
    depends_on = tuple()
    dtype = strax.record_dtype()
    rechunk_on_save = False

    def setup(self):
        utils.experiment = self.config['experiment']

    def chunk_folder(self, chunk_i):
        return os.path.join(self.config['input_dir'], f'{chunk_i:06d}')

    def chunk_paths(self, chunk_i):
        p = self.chunk_folder(chunk_i)
        result = []
        for q in [p + '_pre', p, p + '_post']:
            if os.path.exists(q):
                n_files = len(os.listdir(q))
                if n_files >= self.config['readout_threads']:
                    result.append(q)
                else:
                    print(f'Found incomplete folder {q}: found {n_files} files '
                          f'but expected {self.config["readout_threads"]}. Waiting '
                          f'for more...')
                    result.append(False)
            else:
                result.append(False)
        return tuple(result)

    def source_finished(self):
        end_dir = os.path.join(self.config['input_dir'], 'THE_END')
        if not os.path.exists(end_dir):
            return False
        return len(os.listdir(end_dir)) >= self.config['readout_threads']

    def is_ready(self, chunk_i):
        ended = self.source_finished()
        pre, current, post = self.chunk_paths(chunk_i)
        next_ahead = os.path.exists(self.chunk_folder(chunk_i + 1))
        if (current and (
                (pre and post
                    or chunk_i == 0 and post
                    or ended and (pre and not next_ahead)))):
            return True
        return False

    def load_chunk(self, folder, kind='central'):
        records = np.concatenate([strax.load_file(os.path.join(folder,f),
                                                  compressor='blosc',
                                                  dtype=strax.record_dtype())
                                  for f in os.listdir(folder)])
        records = strax.sort_by_time(records)
        if kind == 'central':
            result = records
        else:
            if self.config['do_breaks']:
                result = strax.from_break(
                    records,
                    safe_break = self.config['safe_break'],
                    left = kind == 'post',
                    tolerant=True)
            else:
                result = records
        result['time'] += self.config['run_start']
        return result

    def compute(self, chunk_i):
        pre, current, post = self.chunk_paths(chunk_i)
        records = np.concatenate(
                ([self.load_chunk(pre, kind='pre')] if pre else [])
                + [self.load_chunk(current)]
                + ([self.load_chunk(post, kind='post')] if post else [])
        )
        strax.baseline(records)
        strax.integrate(records)

        if len(records):
            timespan_sec = (records[-1]['time'] - records[0]['time']) / 1e9
            print(f'{chunk_i}: read {records.nbytes/1e6:.2f} MB '
                  f'({len(records)} records, '
                  f'{timespan_sec:.1f} live seconds)')
        else:
            print(f'{chunk_i}: chunk empty!')

        return records


@export
@strax.takes_config(
        strax.Option('to_pe', track=False,
                     default_by_run=utils.GetGains,
                     help='PMT gains'),
        strax.Option('min_gain', track=False, type=float,
                     default=1e5, help='Minimum PMT gain'),
        strax.Option('hit_threshold', type=int, default=30,
                     help="Hitfinder threshold"),
        strax.Option('left_cut_extension', default=2,
                     help='Cut up to this to many samples before a hit'),
        strax.Option('right_cut_extension', default=15,
                     help='Cut past this many samples after a hit'),
)
class Records(strax.Plugin):
    """
    Shamelessly stolen from straxen
    """
    __version__ = '0.0.2'

    depends_on = ('raw_records',)
    data_kind = 'records'
    compressor = 'zstd'
    parallel = True
    rechunk_on_save = False
    dtype = strax.record_dtype()

    def compute(self, raw_records):
        # Remove records from channels for which the gain is unknown
        # or low
        channels_to_cut = np.argwhere(self.config['to_pe'] > (adc_to_e/self.config['min_gain']))
        r = raw_records
        for ch in channels_to_cut.reshape(-1):
            r = r[r['channel'] != ch]

        strax.zero_out_of_bounds(r)
        hits = strax.find_hits(r, threshold=self.config['hit_threshold'])
        strax.cut_outside_hits(r, hits,
                left_extension = self.config['left_cut_extension'],
                right_extension = self.config['right_cut_extension'])
        return r


@export
@strax.takes_config(
        strax.Option('hit_threshold', type=int, default=30,
                     help="Hitfinder threshold"),
        strax.Option('peak_gap_threshold', type=int, default=150,
                     help='Number of ns without hits to start a new peak'),
        strax.Option('peak_left_extension', type=int, default=20,
                     help='Extend peaks by this many ns to the left'),
        strax.Option('peak_right_extension', type=int, default=120,
                     help='Extend peaks by this many ns to the right'),
        strax.Option('peak_min_chan', type=int, default=1,
                     help='Mininmum number of channels to form a peak'),
        strax.Option('peak_min_area', default=2,
                     help='Minimum area to form a peak'),
        strax.Option('peak_max_duration', default=50e3,
                     help='Maximum peak duration'),
        strax.Option('split_min_height', default=25,
                     help='Minimum prominence height to split peaks'),
        strax.Option('split_min_ratio', default=4,
                     help='Minimum prominence ratio to split peaks'),
        
        strax.Option('split_n_smoothing', default=0,
                     help='how strong the peak smooting is applied (only for find_split_points)'),
                     
        strax.Option('to_pe', track=False,
                     default_by_run=utils.GetGains,
                     help='PMT gains'),
        strax.Option('n_channels', track=False, default_by_run=utils.GetNChan,
                     type=int, help='How many channels'),
)
class Peaks(strax.Plugin):
    """
    Stolen from straxen, extended marginally
    """
    __version__ = "0.0.1F"
    depends_on = ('records',)
    data_kind = 'peaks'
    parallel = True
    rechunk_on_save = True


    def infer_dtype(self):
        dtype_peaks =  strax.peak_dtype(n_channels=8)
        dtype_peaks.append((("time to midpoint (50% quantile)", "time_to_midpoint"), '<i2'))
        
        return(dtype_peaks)
        

    def compute(self, records):
        r = records
        hits = strax.find_hits(r, threshold=self.config['hit_threshold'])
        hits = strax.sort_by_time(hits)

        peaks = strax.find_peaks(hits, self.config['to_pe'],
                                 result_dtype=self.dtype,
                                 gap_threshold=self.config['peak_gap_threshold'],
                                 left_extension=self.config['peak_left_extension'],
                                 right_extension=self.config['peak_right_extension'],
                                 min_channels=self.config['peak_min_chan'],
                                 min_area=self.config['peak_min_area'],
                                 max_duration=self.config['peak_max_duration'],
                                 )
        strax.sum_waveform(peaks, r, adc_to_pe=self.config['to_pe'])
        peaks = peaks[peaks['dt'] > 0]  # removes strange edge case
        peaks = strax.split_peaks(peaks, r, self.config['to_pe'],
                                  min_height=self.config['split_min_height'],
                                  min_ratio=self.config['split_min_ratio'],
                                  n_smoothing=self.config['split_n_smoothing']
                                  )

        strax.compute_widths(peaks)

        return peaks


@export
@strax.takes_config(
        strax.Option('top_pmts', track=False, default=list(range(1,7+1)),
                     type=list, help="Which PMTs are in the top array")
)
class PeakBasics(strax.Plugin):
    """
    Stolen from straxen, extended with risetime. Also replaces
    aft for nonphysical peaks with nan.
    """
    __version__ = "0.0.2B"
    parallel = True
    depends_on = ('peaks',)
    dtype = [
        (('Start time of the peak (ns since unix epoch)',
          'time'), np.int64),
        (('End time of the peak (ns since unix epoch)',
          'endtime'), np.int64),
        (('Peak integral in PE',
            'area'), np.float32),
        (('Number of PMTs contributing to the peak',
            'n_channels'), np.int16),
        (('PMT number which contributes the most PE',
            'max_pmt'), np.int16),
        (('Area of signal in the largest-contributing PMT (PE)',
            'max_pmt_area'), np.int32),
        (('Width (in ns) of the central 50% area of the peak',
            'range_50p_area'), np.float32),
        (('Risetime (in ns) of the peak',
            'risetime'), np.float32),
        (('Time (in ns) to midpoint',
            'time_to_midpoint'), np.float32),
        (('Fraction of area seen by the top array',
            'area_fraction_top'), np.float32),
        (('Length of the peak waveform in samples',
          'length'), np.int32),
        (('Time resolution of the peak waveform in ns',
          'dt'), np.int16),
    ]

    def compute(self, peaks):
        p = peaks
        r = np.zeros_like(p, dtype=self.dtype)
        for q in 'time length dt area'.split():
            r[q] = p[q]
        r['endtime'] = p['time'] + p['dt'] * p['length']
        r['n_channels'] = (p['area_per_channel'] > 0).sum(axis=1)
        r['range_50p_area'] = p['width'][:, 5]
        r['risetime'] = -p['area_decile_from_midpoint'][:,1]
        
        # old buggy version commented out
        # r['time_to_midpoint'] = -p['area_decile_from_midpoint'][:,0]
        # use 50 % quantile time instead
        r['time_to_midpoint'] = -p['time_to_midpoint']
        
        
        r['max_pmt'] = np.argmax(p['area_per_channel'], axis=1)
        r['max_pmt_area'] = np.max(p['area_per_channel'], axis=1)

        area_top = p['area_per_channel'][:, self.config['top_pmts']].sum(axis=1)
        m = p['area'] > 0
        r['area_fraction_top'] = np.full_like(p, fill_value=np.nan, dtype=np.float32)
        r['area_fraction_top'][m] = area_top[m]/p['area'][m]
        return r


@export
@strax.takes_config(
    strax.Option('to_pe', track=False, help='PMT gains',
                     default_by_run=utils.GetGains),
    strax.Option('top_pmts', track=False, default=list(range(1,7+1)),
                 type=list, help="Which PMTs are in the top array"),
    strax.Option('min_reconstruction_area',
                 help='Skip reconstruction if area_top (PE) is less than this',
                 default=100),
    strax.Option('position_weighting_power',
                 help='Weight PMT positions by area seen to this power',
                 default=1.0)
)
class PeakPositionsWeightedSum(strax.Plugin):
    '''
    Position Reconstruction weighted sum
    '''
    __version__ = "0.0.3"
    dtype = [('x_weighted_sum', np.float32,
              'Reconstructed S2 X position (mm) from weighted sum, uncorrected'),
             ('y_weighted_sum', np.float32,
              'Reconstructed S2 Y position (mm) from weighted sum, uncorrected')]
    depends_on = ('peaks',)
    parallel = False

    def setup(self):
        self.pmt_mask = np.zeros_like(self.config['to_pe'], dtype=np.bool)
        self.pmt_mask[self.config['top_pmts']] = self.config['to_pe'][self.config['top_pmts']] > 0
        pmt_x = np.array([-14.,-28,-14.,14.,28.,14.,0.])
        pmt_y = np.array([-28.,0.,28.,28.,0.,-28.,0.])
        self.pmt_positions = np.column_stack((pmt_x, pmt_y))

    def compute(self, peaks):
        # Keep large peaks only
        peak_mask = peaks['area'] > self.config['min_reconstruction_area']
        p = peaks['area_per_channel'][peak_mask, :]
        p = p[:, self.pmt_mask]
        
        # Numpy built in weighted average only works with 1D weights
        # Therefore do it manually
        weights = p ** self.config['position_weighting_power']
        pos = np.nansum(self.pmt_positions[np.newaxis,...] * weights[...,np.newaxis], axis=1)
        pos /=  np.nansum(weights, axis=1)[...,np.newaxis]

        result = np.full_like(peaks, np.nan, dtype=self.dtype)
        result['x_weighted_sum'][peak_mask] = pos[:,0]
        result['y_weighted_sum'][peak_mask] = pos[:,1]
        return result

@export
@strax.takes_config(
    strax.Option('to_pe', track=False, help='PMT gains',
                     default_by_run=utils.GetGains),
    strax.Option('top_pmts', track=False, default=list(range(1,7+1)),
                 type=list, help="Which PMTs are in the top array"),
    strax.Option('min_reconstruction_area',
                 help='Skip reconstruction if area_top (PE) is less than this',
                 default=100),
    strax.Option('nn_model', type=str,
                 help='Filename of the NN for the position-reconstruction.' \
                      'File should be located in "/data/workspace/nn_models/".',
                 default='XeBRA_Position_Reconstruction_NN_Model_DualPhase_7TopPMTs.h5')

)
class PeakPositionsNN(strax.Plugin):
    '''
    Position Reconstruction with neural network

    Version 0.0.1: Weighted Sum
    Version 0.0.2: LRF
    Version 0.0.3: Neural Network

    Status: September 2019, Version 0.0.3

    Position reconstruction for XeBRA using a Deep Feed Forward (DFF) Neural Network
    with Keras trained on Geant4 MC simulations.
    '''
    __version__ = "0.0.3"
    dtype = [('x_nn', np.float32,
              'Reconstructed S2 X position (mm), uncorrected'),
             ('y_nn', np.float32,
              'Reconstructed S2 Y position (mm), uncorrected')]
    depends_on = ('peaks',)
    parallel = True

    def setup(self):
        import keras
        ## PMT mask - select top PMTs
        self.pmt_mask = np.zeros_like(self.config['to_pe'], dtype=np.bool)
        self.pmt_mask[self.config['top_pmts']] = np.ones_like(self.pmt_mask[self.config['top_pmts']])
        ## Load the trained model from corresponding HDF5 file
        self.model_NN = keras.models.load_model(os.path.join('/data/workspace/nn_models', self.config['nn_model']))

    def compute(self, peaks):
        ## Keep large peaks only
        results = np.full_like(peaks, np.nan, dtype=self.dtype)

        for p_i,p in enumerate(peaks):
            apc = p['area_per_channel'][self.pmt_mask]
            if apc.sum() < self.config['min_reconstruction_area']:
                continue
            results[p_i] = tuple(self.reconstructed_position(apc))
        return results

    ## Reconstruct position inside S2 region
    def reconstructed_position(self, input_array):
        ## Normalize sum input to 1 in order to correspond to area fraction in top array
        HFs_input = input_array / np.sum(input_array)
        ## Use model to reconstruct position
        ## Important: Factor 70 for rescaling label
        predictions = self.model_NN.predict(np.array([HFs_input]))[0]*70
        return predictions

@export
@export
@strax.takes_config(
    strax.Option("default_posrec_algorithm",
                 help="default reconstruction algorithm that provides (x,y)",
                 default="nn",
                 )
)
class PeakPositions(strax.Plugin):
    '''
    Provides default positions to use for further processing.

    Selects one of the position reconstrution plugins and uses its output
    to provide 'x' and 'y', for use in further straxbra plugins.
    '''
    __version__ = "0.1.0"
    dtype = [('x', np.float32,
              'Reconstructed S2 X position (mm), uncorrected'),
             ('y', np.float32,
              'Reconstructed S2 Y position (mm), uncorrected')]
    depends_on = ('peak_positions_nn','peak_positions_weighted_sum')
    parallel = False

    def compute(self, peaks):
        algorithm = self.config['default_posrec_algorithm']
        result = np.full_like(peaks, np.nan, dtype=self.dtype)
        result['x'] = peaks[f'x_{algorithm}']
        result['y'] = peaks[f'y_{algorithm}']
        return result



@export
@strax.takes_config(
    strax.Option('s1_max_width', default=150,
                 help="Maximum (IQR) width of S1s"),
    strax.Option('s1_min_n_channels', default=1,  # TODO improve
                 help="Minimum number of PMTs that must contribute to a S1"),
    strax.Option('s2_min_area', default=10,
                 help="Minimum area (PE) for S2s"),
    strax.Option('s2_min_width', default=200,
                 help="Minimum width for S2s"))
class PeakClassification(strax.Plugin):
    __version__ = '0.0.1'
    depends_on = ('peak_basics',)
    dtype = [
        ('type', np.int8, 'Classification of the peak.')]
    parallel = True

    def compute(self, peaks):
        p = peaks
        r = np.zeros(len(p), dtype=self.dtype)

        is_s1 = p['n_channels'] >= self.config['s1_min_n_channels']
        is_s1 &= p['range_50p_area'] < self.config['s1_max_width']
        r['type'][is_s1] = 1

        is_s2 = p['area'] > self.config['s2_min_area']
        is_s2 &= p['range_50p_area'] > self.config['s2_min_width']
        r['type'][is_s2] = 2

        return r



@export
# @strax.takes_config(
#         strax.Option('min_s2_area_se', default=1e4,
#                      type=float, help="Min energy for a S2")
# )
class PeakIsolations(strax.Plugin):
    """
    For Single Electron Analysis.
    """
    __version__ = "0.0.1"
    parallel = True
    save_when = strax.SaveWhen.EXPLICIT
    depends_on = ('peak_basics', )  # 'peak_positions', )
    dtype = [
        (('Start time of the peak (ns since unix epoch)',
          'time'), np.int64),
        (('End time of the peak (ns since unix epoch)',
          'endtime'), np.int64),
        (('Time since last peak',
            'iso_left'), np.float32),
        (('Time until next peak',
            'iso_right'), np.float32),
        (('Minimum of iso_right and iso_left',
            'iso_min'), np.float32),
        (('Bool array of selected single electrons',
            'se_selection'), np.bool_),
        (('Time since previous S2 from selection for single electron selection, else NaN',
            'time_since_s2'), np.float32),
        (('SE x-position taken from previous large S2',
            'x_se'), np.float32),
        (('SE y-position taken from previous large S2',
            'y_se'), np.float32),
    ]

    def compute(self, peaks):
        r = np.full_like(peaks, np.nan, dtype=self.dtype)

        r['time'] = peaks['time']
        r['endtime'] = peaks['endtime']

        r['iso_left'] = r['time'] - shift(r['endtime'], 1, cval=np.nan)
        r['iso_right'] = shift(r['time'], -1, cval=np.nan) - r['endtime']
        r['iso_left'][0] = 0
        r['iso_right'][-1] = 0
        r['iso_min'] = np.minimum(r['iso_left'], r['iso_right'])

        ### further extension would be to take all peaks within
        ### a certain time window after a big s2 as potential SEs
        ### functionality of "after_s2_mask"-function in peaktools.py
        ### can be used.

        # s2_selection = peaks['area'] > self.config['min_s2_area_se']

        return r



@strax.takes_config(
    strax.Option('min_area_fraction', default=0.5,
                 help='The area of competing peaks must be at least '
                      'this fraction of that of the considered peak'),
    strax.Option('nearby_window', default=int(5e3),
                 help='Peaks starting within this time window (on either side)'
                      'in ns count as nearby.'),
)
class NCompeting(strax.OverlapWindowPlugin):
    depends_on = ('peak_basics',)
    dtype = [
        (('Number of nearby larger or slightly smaller peaks',
            'n_competing'), np.int32),]

    def get_window_size(self):
        return 2 * self.config['nearby_window']

    def compute(self, peaks):
        results = n_competing=self.find_n_competing(
            peaks,
            window=self.config['nearby_window'],
            fraction=self.config['min_area_fraction'])
        return dict(n_competing=results)

    @staticmethod
    @numba.jit(nopython=True, nogil=True, cache=True)
    def find_n_competing(peaks, window, fraction):
        n = len(peaks)
        t = peaks['time']
        a = peaks['area']
        results = np.zeros(n, dtype=np.int32)

        left_i = 0
        right_i = 0
        for i, peak in enumerate(peaks):
            while t[left_i] + window < t[i] and left_i < n - 1:
                left_i += 1
            while t[right_i] - window < t[i] and right_i < n - 1:
                right_i += 1
            results[i] = np.sum(a[left_i:right_i + 1] > a[i] * fraction)

        return results - 1


@export
@strax.takes_config(
    strax.Option('trigger_min_area', default=100,
                 help='Peaks must have more area (PE) than this to '
                      'cause events'),
    strax.Option('trigger_max_competing', default=7,
                 help='Peaks must have FEWER nearby larger or slightly smaller'
                      ' peaks to cause events'),
    strax.Option('left_event_extension', default=int(50e3),
                 help='Extend events this many ns to the left from each '
                      'triggering peak'),
    strax.Option('right_event_extension', default=int(50e3),
                 help='Extend events this many ns to the right from each '
                      'triggering peak'),
    strax.Option('max_event_duration', default=int(100e3),
                 help='Events longer than this are forcefully ended, '
                      'triggers in the truncated part are lost!'),
)
class Events(strax.OverlapWindowPlugin):
    depends_on = ['peak_basics', 'n_competing']
    data_kind = 'events'
    dtype = [
        (('Event number in this dataset',
            'event_number'), np.int64),
        (('Event start time in ns since the unix epoch',
            'time'), np.int64),
        (('Event end time in ns since the unix epoch',
            'endtime'), np.int64)]
    events_seen = 0
    rechunk_on_save = False

    def get_window_size(self):
        return (2 * self.config['left_event_extension'] +
                self.config['right_event_extension'])

    def compute(self, peaks):
        le = self.config['left_event_extension']
        re = self.config['right_event_extension']

        triggers = peaks[
            (peaks['area'] > self.config['trigger_min_area'])
            & (peaks['n_competing'] <= self.config['trigger_max_competing'])]

        # Join nearby triggers
        t0, t1 = strax.find_peak_groups(
            triggers,
            gap_threshold=le + re + 1,
            left_extension=le,
            right_extension=re,
            max_duration=self.config['max_event_duration'])

        result = np.zeros_like(t0, dtype=self.dtype)
        result['time'] = t0
        result['endtime'] = t1
        result['event_number'] = np.arange(len(result)) + self.events_seen

        self.events_seen += len(result)

        return result
        # TODO: someday investigate if/why loopplugin doesn't give
        # anything if events do not contain peaks..


@export
class EventBasics(strax.LoopPlugin):
    __version__ = '0.0.2'
    depends_on = ('events',
                  'peak_basics', 'peak_classification',
                  'peak_positions', 'n_competing')

    def infer_dtype(self):
        dtype = [(('Number of peaks in the event',
                   'n_peaks'), np.int32),
                 (('Drift time between main S1 and S2 in ns',
                   'drift_time'), np.int64)]
        for i in [1, 2]:
            dtype += [((f'Main S{i} peak index',
                        f's{i}_index'), np.int32),
                      ((f'Main S{i} area (PE), uncorrected',
                        f's{i}_area'), np.float32),
                      ((f'Main S{i} area (PE), uncorrected, bottom PMTs only',
                        f's{i}_area_b'), np.float32),
                      ((f'Main S{i} area fraction top',
                        f's{i}_area_fraction_top'), np.float32),
                      ((f'Main S{i} width (ns, 50% area)',
                        f's{i}_range_50p_area'), np.float32),
                      ((f'Main S{i} time to midpoint (ns)',
                        f's{i}_time_to_midpoint'), np.float32),  
                      ((f'Main S{i} number of competing peaks',
                        f's{i}_n_competing'), np.int32)]
        dtype += [(f'x_s2', np.float32,
                   f'Main S2 reconstructed X position (cm), uncorrected',),
                  (f'y_s2', np.float32,
                   f'Main S2 reconstructed Y position (cm), uncorrected',)]
        dtype += [(f's2_largest_other', np.float32,
                   f'Area of largest other s2'),
                  (f's1_largest_other', np.float32,
                   f'Area of largest other s1')
                  ]
        return dtype

    def compute_loop(self, event, peaks):
        result = dict(n_peaks=len(peaks))
        if not len(peaks):
            return result

        main_s = dict()
        for s_i in [2, 1]:
            s_mask = peaks['type'] == s_i

            # For determining the main S1, remove all peaks
            # after the main S2 (if there was one)
            # This is why S2 finding happened first
            if s_i == 1 and result[f's2_index'] != -1:
                s_mask &= peaks['time'] < main_s[2]['time']

            ss = peaks[s_mask]
            s_indices = np.arange(len(peaks))[s_mask]

            if not len(ss):
                result[f's{s_i}_index'] = -1
                continue

            main_i = np.argmax(ss['area'])

            if ss['n_competing'][main_i]>0 and len(ss['area'])>1:
                other_i = np.argsort(ss['area'])[-2]
                result[f's{s_i}_largest_other'] = ss['area'][other_i]

            result[f's{s_i}_index'] = s_indices[main_i]
            s = main_s[s_i] = ss[main_i]

            for prop in ['area', 'area_fraction_top',
                         'range_50p_area', 'n_competing','time_to_midpoint']:
                result[f's{s_i}_{prop}'] = s[prop]
            if s_i == 2:
                for q in 'xy':
                    result[f'{q}_s2'] = s[q]

        # Compute a drift time only if we have a valid S1-S2 pairs
        if len(main_s) == 2:
            result['drift_time'] = main_s[2]['time'] - main_s[1]['time']

        return result


@export
class EventKryptonBasics(strax.LoopPlugin):
    """Stolen from EventBasics and modified."""
    __version__ = '0.0.1'
    depends_on = ('events',
                  'peak_basics', 'peak_classification',
                  'peak_positions', 'n_competing')

    def infer_dtype(self):
        dtype = [(('Time of second largest S1 relative to main S1 in ns',
                   't_rel_second_s1'), np.int32)]

        for s_i in [1,2]:
            dtype += [((f'Number of PMTs contributing to main S{s_i}',
                        f's{s_i}_n_channels'), np.int16),
                      ((f'Width (in ns) of the central 50% area of other largest S{s_i}',
                        f'other_s{s_i}_range_50p_area'), np.float32),
                      ((f'Number of PMTs contributing to other largest S{s_i}',
                        f'other_s{s_i}_n_channels'), np.int16),
                      ((f'Largest other S{s_i} peak index',
                        f'other_s{s_i}_index'), np.int32)
                     ]

        return dtype

    def compute_loop(self, event, peaks):
        result = {'t_rel_second_s1' : 0}
        if not len(peaks):
            return result

        main_s = dict()
        index = dict()
        for s_i in [2, 1]:
            s_mask = peaks['type'] == s_i

            # For determining the main S1, remove all peaks
            # after the main S2 (if there was one)
            # This is why S2 finding happened first
            if s_i == 1 and index['s2'] != -1:
                s_mask &= peaks['time'] < main_s[2]['time']

            ss = peaks[s_mask]
            s_indices = np.arange(len(peaks))[s_mask]

            if not len(ss):
                index[f's{s_i}'] = -1
                continue

            main_i = np.argmax(ss['area'])

            if ss['n_competing'][main_i]>0 and len(ss['area'])>1:
                other_i = np.argsort(ss['area'])[-2]
                result[f'other_s{s_i}_n_channels'] = ss['n_channels'][other_i]
                result[f'other_s{s_i}_range_50p_area'] = ss['range_50p_area'][other_i]
                result[f'other_s{s_i}_index'] = s_indices[other_i]

                if s_i == 1:
                    result['t_rel_second_s1'] = ss['time'][other_i] - ss['time'][main_i]



            index[f's{s_i}'] = s_indices[main_i]
            s = main_s[s_i] = ss[main_i]

            result[f's{s_i}_n_channels'] = s['n_channels']

        return result




@export
@strax.takes_config(
    strax.Option('sp_krypton_s1_area_min', default=30,
                 help='minimum area for a peak to potentially be a S1'),
    strax.Option('sp_krypton_s1_area_max', default=800,
                 help='maximum area for a peak to potentially be a S1'),
    strax.Option('sp_krypton_s1s_dt_max', default=2500,
                 help='maximum time difference beetween 2 peaks '
                      'to be considered two S1s'),
    
    strax.Option('sp_krypton_electron_lifetime', default=-1,
                 help='electron lifetime in this run [µs)'),
                 
    strax.Option('sp_krypton_min_S2_area', default=100,
                 help='Minimum S2 area (in PE)'),
    strax.Option('sp_krypton_max_S2_area', default=1_000_000,
                 help='Maximum S2 area (in PE)'),

    strax.Option('sp_krypton_min_drifttime_ns', default=0,
                 help='Minimum Drifttime (ns)'),
    strax.Option('sp_krypton_max_drifttime_ns', default=40_000,
                 help='Maximum drifttime (ns)'),
)

@export
class EventsSinglePhaseKryptonBasics(strax.LoopPlugin):
    """Stolen from Events_Krypton_basics and modified."""
    __version__ = '0.0.1'
    depends_on = ('events',
                  'peak_basics', 'peak_classification',
                  'peak_positions', 'n_competing')

    def infer_dtype(self):
        dtype = [
                (('first/last peaks start time',
                   'first_and_last_peak_time'), np.int64, (2, )),
                
                (('first S1 like peak area in PE',
                   's1_1_area'), np.float32),
                (('second S1 like peak area in PE',
                   's1_2_area'), np.float32),
                (('first S1 like peak width in PE',
                   's1_1_width'), np.float32),
                (('second S1 like peak width in PE',
                   's1_2_width'), np.float32),
                (('time difference of s1 like peaks',
                   'dt_s1s'), np.float32),
                (('area ratio of both s1 likes',
                   'ratio_area_s1s'), np.float32),
                
                (('time of first s1 like',
                   's1_1_time'), np.int64, ),
                (('time of second s1 like',
                   's1_2_time'), np.int64, ),
                
                
                
                
                
                (('id of s1 and largest peaks',
                   'peaks_ids'), np.int64,(5,)),
                
                # those entries are only here to keep my old script working
                # please don't use those entries as their label might be ambiguous
                
                (('largest non s1-like peak area in PE',
                   'largest_peak_area'), np.float32),
                (('largest non s1-like peak width in ns',
                   'largest_peak_width'), np.float32),
                (('time difference of first s1 like to largest non s1-like peak',
                   'dt_s11_largest'), np.float32),
                
                (('largest non peak area in PE',
                   'largest_peak_all_area'), np.float32),
                (('largest peak width in ns',
                   'largest_peak_all_width'), np.float32),
                (('time difference of first s1 like to largest peak',
                   'dt_s11_largest_peak_all'), np.float32),
                
                (('largest peak is not S1 like',
                   'largest_peak_is_not_S1_like'), np.bool),
                
                (('second largest non s1-like peak area in PE',
                   'second_largest_peak_area'), np.float32),
                (('second largest non s1-like width in ns',
                   'second_largest_peak_width'), np.float32),
                (('time difference of largest non s1 like peaks',
                   'dt_largest'), np.float32),
                
                
                # new: actual S2 peaks-data (same as above but labeld better)
                (('S2 area (in PE, if S2 exists)',
                   'S2_area'), np.float32),
                (('corrected S2 area (in PE, if S2 exists)',
                   'S2_area_corr'), np.float32),                   
                (('S2 width (ns, if S2 exists))',
                   'S2_width'), np.float32),
                (('drifttime in µs',
                   'S2_dt'), np.float32),
                
                
                
                (('pulses before the first S1 (area, width, time, n_channels), only first 10',
                   'S2_prepulses'), np.float32, (10, 4)),
                
                (('pulses between the second S1 and the S2 (area, width, time, n_channels), only first 5',
                   'S2_midpulses'), np.float32, (5, 4)),
                
                (('pulses after the S2 (area, width, time, n_channels), only first 20',
                   'S2_afterpulses'), np.float32, (20, 4)),
                
                (('Number of pulses before the first S1',
                   'S2_N_prepulses'), np.int64),
                   
                (('Number of pulses between the second S1 and the S2',
                   'S2_N_midpulses'), np.int64),
                
                (('Number of pulses after the S2',
                   'S2_N_afterpulses'), np.int64),
                
                
                
                   
                ]

        return dtype

    def compute_loop(self, event, peaks):
        result = {}
        if not len(peaks):
            return result

        result["first_and_last_peak_time"] = [peaks["time"][0], peaks["time"][-1]]
        
        # all the s1 like oprtations
        peak_ids_ss = np.nonzero(((peaks["area"] > self.config["sp_krypton_s1_area_min"]) & (peaks["area"] < self.config["sp_krypton_s1_area_max"])))[0]
        peaks_ss = peaks[peak_ids_ss]
        
        diff_time = np.array([
                p2["time"]-p1["time"]
                for p1, p2
                in zip(peaks_ss[:-1], peaks_ss[1:])
        ])
        # s1s must be between 2500 ns (~16 half-lifes)
        ids_s1_test = np.nonzero(diff_time < self.config["sp_krypton_s1s_dt_max"])[0]
        if len(ids_s1_test) < 1:
            return(result)
         
        
        id_first_s1  = peak_ids_ss[min(ids_s1_test)]
        id_second_s1  = peak_ids_ss[min(ids_s1_test) + 1]


        result["s1_1_time"] = peaks[id_first_s1]["time"]
        result["s1_2_time"] = peaks[id_second_s1]["time"]        
        result["s1_1_area"] = peaks[id_first_s1]["area"]
        result["s1_2_area"] = peaks[id_second_s1]["area"]
        result["s1_1_width"] = peaks[id_first_s1]["range_50p_area"]
        result["s1_2_width"] = peaks[id_second_s1]["range_50p_area"]
        result["dt_s1s"] = peaks[id_second_s1]["time"]- peaks[id_first_s1]["time"]
        result["ratio_area_s1s"] = peaks[id_first_s1]["area"] / peaks[id_second_s1]["area"]
        
        peak_ids_by_area = np.argsort(peaks["area"])
        id_largest_peak_all = peak_ids_by_area[-1]
        
        
        result["largest_peak_all_area"] = peaks[id_largest_peak_all]["area"]
        result["largest_peak_all_width"] = peaks[id_largest_peak_all]["range_50p_area"]
        result["dt_s11_largest_peak_all"] = peaks[id_largest_peak_all]["time"] - peaks[id_first_s1]["time"]
        
        
        
        peak_ids_by_area = [id for id in peak_ids_by_area if id not in [id_first_s1, id_second_s1]]
    
        result["peaks_ids"] = [id_first_s1, id_second_s1, -1, -1, id_largest_peak_all]
    
        if len(peak_ids_by_area) < 1:
            return(result)
        
        
        id_largest_s = peak_ids_by_area[-1]
        
        # maybe the seconds s1 is larger than the first one....
        result["largest_peak_is_not_S1_like"] = (id_largest_peak_all not in [id_first_s1, id_second_s1])
        
        
        
        
        result["dt_s11_largest"] = peaks[id_largest_s]["time"] - peaks[id_first_s1]["time"]
        result["largest_peak_area"] = peaks[id_largest_s]["area"]
        result["largest_peak_width"] = peaks[id_largest_s]["range_50p_area"]
        
        
        
        # check if S2-like might be S2:
        if (
                (result["dt_s11_largest"] >= self.config["sp_krypton_min_drifttime_ns"])
            and (result["dt_s11_largest"] <= self.config["sp_krypton_max_drifttime_ns"])
            and (result["largest_peak_area"] >= self.config["sp_krypton_min_S2_area"])
            and (result["largest_peak_area"] <= self.config["sp_krypton_max_S2_area"])
            and (result["largest_peak_is_not_S1_like"])
            and (id_largest_s > id_second_s1) # S2 MUST come after second S1
        ):
            result["S2_area"] = peaks[id_largest_s]["area"]
            result["S2_width"] = peaks[id_largest_s]["range_50p_area"]
            result["S2_dt"] = result["dt_s11_largest"]  / 1000
            
            
        # prepulses
        result["S2_prepulses"] = [(0,0,0,0)]*10
        result["S2_N_prepulses"] = id_first_s1
        for i, peak_id in enumerate(range(0, id_first_s1)):
            if i == 10:
                break
            result["S2_prepulses"][i] = (
            
                peaks[peak_id]["area"],
                peaks[peak_id]["range_50p_area"],
                max(0, peaks[peak_id]["time"] - peaks[peak_id-1]["time"]),
                peaks[peak_id]["n_channels"],
            )
        

        # midpulses
        result["S2_midpulses"] = [(0,0,0,0)]*5
        result["S2_N_midpulses"] = id_largest_s - id_second_s1 - 1
        for i, peak_id in enumerate(range(id_second_s1+1, id_largest_s)):
            if i == 5:
                break
            result["S2_midpulses"][i] = (
            
                peaks[peak_id]["area"],
                peaks[peak_id]["range_50p_area"],
                peaks[peak_id]["time"] - peaks[peak_id-1]["time"],
                peaks[peak_id]["n_channels"],
            )
                



        # Afterpulses
        result["S2_afterpulses"] = [(0,0,0,0)]*20
        result["S2_N_afterpulses"] = len(peaks) - id_largest_s - 1
        for i, peak_id in enumerate(range(id_largest_s+1, len(peaks)-1)):
            if i == 20:
                break

            result["S2_afterpulses"][i] = (
            
                peaks[peak_id]["area"],
                peaks[peak_id]["range_50p_area"],
                peaks[peak_id]["time"] - peaks[peak_id-1]["time"],
                peaks[peak_id]["n_channels"],
            )
            
        
        
        

        if len(peak_ids_by_area) == 1:
            result["peaks_ids"] = [id_first_s1, id_second_s1, id_largest_s, -1, id_largest_peak_all]
            return(result)
        
        
        id_second_largest_s = peak_ids_by_area[-2]
        result["peaks_ids"] = [id_first_s1, id_second_s1, id_largest_s, id_second_largest_s, id_largest_peak_all]
        result["second_largest_peak_area"] = peaks[id_second_largest_s]["area"]
        result["second_largest_peak_width"] = peaks[id_second_largest_s]["range_50p_area"]
        result["dt_largest"] = peaks[id_largest_s]["time"] - peaks[id_second_largest_s]["time"]
        
        
        return result











# Other settigns are already defined for EventsSinglePhaseKryptonBasics
@export
@strax.takes_config(
    strax.Option('sp_krypton_s1_area_min', default=25,
                 help='minimum area for a peak to potentially be a S1'),
    strax.Option('sp_krypton_s1_area_max', default=400,
                 help='maximum area for a peak to potentially be a S1'),
    strax.Option('sp_krypton_s1s_dt_max', default=1500,
                 help='maximum time difference beetween 2 peaks'
                      'to be considered two S1s'),
    strax.Option('sp_krypton_dt_s1s_s2s_max', default=50,
                 help='how much the S2s are allowed to be further aparth than the S1s'
                      'to be considered two S1s'),
    strax.Option('sp_krypton_min_drifttime_ns', default=0,
                 help='Minimum Drifttime (ns)'),
    strax.Option('sp_krypton_max_drifttime_ns', default=500_000,
                 help='Maximum drifttime (ns)'),
)



@export
class SpKrypton(strax.LoopPlugin):
    """
    New and improved version for single phase Krypton data
    optimiced for aggressive cutting: min_height = 0
    
    """
    __version__ = '0.0.4.29'
    depends_on = ('events', 'peaks', 'peak_basics')
  
  
  
    def infer_dtype(self):
        dtype = [


                # developer info
                (('DEVELOPER: number of peaks that might me S1 or S2',
                   'DEVELOPER_n_pot_signals'), np.int16, 2),
                                # developer info
                (('DEVELOPER: at which stage the compute loop exits',
                   'DEVELOPER_fails'), np.bool, 16),
                                # developer info
                                # we just use 9, but i guess its using 2 bits anyway,
                                # so why not just future proofing this....


                # general Info about event
                # helps to get all peaks and plot them
                (('timestamp of the first peak in the event (ns)',
                   'time_event'), np.int64),
                (('number of peaks in the event',
                   'n_peaks'), np.int16),
                (('number of large peaks in the event (larger than sp_krypton_s1_area_min)',
                   'n_peaks_large'), np.int16),
                (('timestamps of first 20 large peaks',
                   'time_large_peaks'), np.int64, 20),
                (('midpoint times of first 20 large peaks',
                   'time_mp_large_peaks'), np.int64, 20),
                   
                   
                # simple booleans to base selection on
                (('wheter the event is a krypton event (2 S1s + 1 or 2 S2s)',
                   'is_event'), np.bool),
                (('wheter the S1s got split or not',
                   's1_split'), np.bool),
                (('wheter the S2s got split or not',
                   's2_split'), np.bool),
                (('wheter minimum peaks are found',
                   'min_peaks'), np.bool),

                # helps to plot only the signals
                (('timestamps of S11, S12, S21 (and S22 if it is found) (ns)',
                   'time_signals'), np.int64, 4),
                
                # used to calculate time differences
                (('center of peaks (ns) relative to first S1',
                   'time_peaks'), np.int64, 4),
                   
                   
                (('the entire waveform of the peaks for convientent and quick access',
                   'data_peaks'), np.float32, (4, 200)),
                
                (('Time resolution of the peaks waveform in ns',
                  'dt'), np.int16, 4),
                
                
                # uncorrected + corrected areas of individual peaks
                (('S11 area (PE)',
                   'area_s11'), np.float32),
                (('corrected S11 area (PE)',
                   'cS11'), np.float32),
                (('S12 area (PE)',
                   'area_s12'), np.float32),
                (('corrected S12 area (PE)',
                   'cS12'), np.float32),
                
                (('S21 area (PE)',
                   'area_s21'), np.float32),
                (('corrected S21 area (PE)',
                   'cS21'), np.float32),
                (('S22 area (PE)',
                   'area_s22'), np.float32),
                (('corrected S22 area (PE)',
                   'cS22'), np.float32),
                   
                # uncorrected + corrected areas of combined S1 and S2
                (('S1 area (PE)',
                   'area_s1'), np.float32),
                (('corrected S1 area (PE)',
                   'cS1'), np.float32),
                (('S2 area (PE)',
                   'area_s2'), np.float32),
                (('corrected S2 area (PE)',
                   'cS2'), np.float32),
                
                # widths of individual peaks and Signals
                (('Width of S11 (ns)',
                   'width_s11'), np.float32),
                (('Width of S12 (ns)',
                   'width_s12'), np.float32),
                (('Width of S21 (ns)',
                   'width_s21'), np.float32),
                (('Width of S22 (ns)',
                   'width_s22'), np.float32),
                (('Width of S1 (ns)',
                   'width_s1'), np.float32),
                (('Width of S2 (ns)',
                   'width_s2'), np.float32),
                   
                # Krypton decay times
                (('time from S11 to S12 (ns)',
                   'time_decay_s1'), np.int64),
                (('time from S21 to S22 (ns)',
                   'time_decay_s2'), np.int64),
                
                # Drifttime
                (('drifttime between S11 and S21 (µs)',
                   'time_drift'), np.float32),
                (('drifttime between S12 and S22 (µs)',
                   'time_drift2'), np.float32),
                
                
                # create those values so we can populate them easyil 
                (('corrected drifttime between S11 and S21 (µs)',
                   'cdt'), np.float32),
                (('corrected drifttime between S12 and S22 (µs)',
                   'cdt2'), np.float32),
                (('z-position of event based on S11 and S21 (mm)',
                   'z'), np.float32),
                (('z-position of event based on S12 and S22 (mm)',
                   'z2'), np.float32),
                
                
                
                # All the energies (if g1, g2, elifetime are provided)
                (('Energy of S1 (keV)',
                   'energy_s1'), np.float32),
                (('Energy of S2 (keV)',
                   'energy_s2'), np.float32),
                (('Total Energy of Event (keV)',
                   'energy_total'), np.float32),
                
                   
                ]

        return dtype

    def store_peak(self, p, i, result):
        f = ["s11", "s12", "s21", "s22"][i]
        
        result[f"area_{f}"] = p["area"]
        result[f"width_{f}"] = p["range_50p_area"]
        result["time_signals"][i] = p["time"]
        
        result["dt"][i] = p["dt"]
        result["data_peaks"][i] = p["data"]
        
        if i == 0:
            result["time_peaks"][i] = p["time_to_midpoint"]
        else:
            result["time_peaks"][i] = p["time_to_midpoint"] + (p["time"]-result["time_signals"][0])
        return(None)


    def compute_loop(self, event, peaks):
        
        result = {}
        result["n_peaks"] = len(peaks)
        result["time_event"] = peaks[0]["time"]
        # initialize multipeak variables
        result["DEVELOPER_fails"] = [False]*16
        result["data_peaks"] = [[0]*200]*4
        result["time_signals"] = [-1]*4
        result["time_peaks"] = [-1]*4
        result["dt"] = [-1]*4
        
        
        if len(peaks) == 0:
            result["DEVELOPER_fails"][0] = True
            return(result)
        
        peaks_large = peaks[peaks["area"] >= self.config["sp_krypton_s1_area_min"]]
        
        
        # export large peaks for easy investigations
        result["n_peaks_large"] = len(peaks_large)
        result["time_large_peaks"] = [-1]*20
        result["time_mp_large_peaks"] = [-1]*20
        for i, p in enumerate(peaks_large[:20]):
            result["time_large_peaks"][i] = p["time"]
            result["time_mp_large_peaks"][i] = p["time_to_midpoint"]
        
        
        
        # move all S1 stuff up here to access it even if there are not enough events
        # if we have at least one peaks this  is stored as first S1
        S11 = peaks_large[0]
        self.store_peak(p = S11, i = 0, result = result)
        
        # fallback for later
        result["area_s1"] = S11["area"]
        result["width_s1"] = S11["range_50p_area"]
        # if S1 is not split the S2s are definitely not split
        # here we force test later to definitely fail
        decay1 = -2*self.config["sp_krypton_s1s_dt_max"]
        
        
        if S11["area"] > self.config["sp_krypton_s1_area_max"]:
            result["DEVELOPER_fails"][3] = True
        
        if len(peaks_large) < 2:
            result["DEVELOPER_fails"][1] = True
            return(result)
        
        # if we also have second peak that might be our second S1
        S12 = peaks_large[1]
        self.store_peak(p = S12, i = 1, result = result)
        decay1 = result["time_peaks"][1] - result["time_peaks"][0]
        result["time_decay_s1"] = decay1
        
        if S12["area"] > self.config["sp_krypton_s1_area_max"]:
            result["DEVELOPER_fails"][4] = True
        if (S12["time"] - S11["time"]) > self.config["sp_krypton_s1s_dt_max"]:
            result["DEVELOPER_fails"][5] = True
        if S11["area"] < S12["area"]:
            result["DEVELOPER_fails"][8] = True
        
        
        if (
                (result["DEVELOPER_fails"][5] is True)
            ):
            pass
            # peak might be S2 instead of S1
            # come up with logic to test which peak is what
        
        
        else:
            # combined S1 
            result["area_s1"] = S11["area"] + S12["area"]
            result["width_s1"] = S11["range_50p_area"] + S12["range_50p_area"]
            # store first decay for later use 
        
        
        
        
        
        
        # ALL S1 STUFF DONE
        if len(peaks_large) < 3:
            # not enough peaks
            result["DEVELOPER_fails"][2] = 1
            return(result)
        
        
        # if a third peak exits it might be our first (or combined) S2
        S21 = peaks_large[2]
        self.store_peak(p = S21, i = 2, result = result)
        
        if (S21["time"] - S11["time"]) > self.config["sp_krypton_max_drifttime_ns"]:
            result["DEVELOPER_fails"][6] = True
        if S21["area"] < S11["area"]:
            result["DEVELOPER_fails"][7] = True
        result["time_drift"] = (result["time_peaks"][2] - result["time_peaks"][0])/1000
        
        # use first S2 area as total S2 area just as fallback if there are no more peaks
        result["area_s2"] = result["area_s21"]
        result["width_s2"] = result["width_s21"]
        
        
        
        # check if S22 exits (just assume its the third peak
        if len(peaks_large) > 3:
            S22 = peaks_large[3]
            self.store_peak(p = S22, i = 3, result = result)
            decay2 = result["time_peaks"][3] - result["time_peaks"][2]
            result["time_decay_s2"] = decay2
            result["time_drift2"] = (result["time_peaks"][3] - result["time_peaks"][1])/1000
            
            if np.abs(decay2 - decay1) <= self.config["sp_krypton_dt_s1s_s2s_max"]:
                result["s2_split"] = True
                result["area_s2"] = result["area_s21"] + result["area_s22"]
                result["width_s2"] = result["width_s21"] + result["width_s22"]
        
                # only 
                if len(peaks_large) == 4:
                    result["min_peaks"] = True
        else:
            # here we have only exact 3 peaks, which is the minimum number for an unsplit event
            result["min_peaks"] = True        
        
        
        # multiple things to check 
        
        # all criteria for an event have been satisfied
        mask_ = [True]*16
        if True not in result["DEVELOPER_fails"]:
            result["is_event"] = True
        return(result)




@export
@strax.takes_config(
    strax.Option('sp_krypton_s1_area_min', default=25,
                 help='minimum area for a peak to potentially be a S1'),
    strax.Option('sp_krypton_max_drifttime_ns', default=500_000,
                 help='Maximum drifttime (ns)'),
)
@export
class GaussfitPeaks(strax.Plugin):
    """
    Stolen from straxen, extended with risetime. Also replaces
    aft for nonphysical peaks with nan.
    """
    __version__ = "0.0.0.3B"
    parallel = True
    depends_on = ('peaks',)
    
    dtype_add = [
        (('fit result single gaus (mu, sigma, A)',
          'fit_s'), np.float32, 3),
        (('unceretainty of fit result single gaus (s_mu, s_sigma, s_A)',
          'sfit_s'), np.float32, 3),
        (('fit result single gaus 2x(mu, sigma, A)',
          'fit_d'), np.float32, 6),
        (('unceretainty of fit result single gaus 2x(s_mu, s_sigma, s_A)',
          'sfit_d'), np.float32, 6),
    ]

    def infer_dtype(self):
        dtype_peaks =  strax.peak_dtype(n_channels=8)
        dtype_peaks.append((("time to midpoint (50% quantile)", "time_to_midpoint"), '<i2'))
        for x in self.dtype_add:
            dtype_peaks.append(x)
        
        return(dtype_peaks)
    
    
    def sg(self, x, mu, s, A):
        return(
            A * np.exp(-((x-mu)**2/(2*s**2)))
        )

    def dg(self, x, mu1, s1, A1, mu2, s2, A2):
        return(
              A1 * np.exp(-((x-mu1)**2/(2*s1**2)))
            + A2 * np.exp(-((x-mu2)**2/(2*s2**2)))
        )
        
    def fit_gausses(self, p, f = "sg"):
        x = np.arange(p["length"])*p["dt"]
        y = p["data"][:p["length"]]
        
        if f == "dg":
            #pars = ["mu1", "s1", "A1", "mu2", "s2", "A2"]
            p0 = [x[np.argmax(y)], max(x)/20, max(y), x[np.argmax(y)]+500, max(x)/20, max(y)/2]
            f = self.dg
        else:
            #pars = ["mu", "s", "A"]
            p0 = [x[np.argmax(y)], max(x)/10, max(y)]
            f = self.sg
        fit, cov = curve_fit(
            f,
            x, y,
            p0 = p0
        )
        sfit = np.diag(cov)**.5
        
        return(fit, sfit)
    
    iteraton = 0
    def compute(self, peaks):
        ps = peaks[peaks["area"] >= self.config["sp_krypton_s1_area_min"]]
        idx = np.nonzero(np.diff(ps["time"]) <= self.config["sp_krypton_max_drifttime_ns"])[0]
        idx = np.unique(np.append(idx, idx+1))
        ps = ps[idx]
        
        
        lp = len(ps)
        slp = len(str(lp))
        
        r = np.zeros_like(ps, dtype=self.dtype)
        for field in ps.dtype.names:
            r[field] = ps[field]

        self.iteraton += 1
        print(f"\nchunk: {self.iteraton}")
        print(f"min peak time: {min(r['time'])}")
        print(f"keeping {lp} / {len(peaks)} peaks")

        for i, p in enumerate(ps):
            if i%50 == 0:
                print(f"\r  {i:>{slp}}/{len(ps):>{slp}} ", end = "")
            
            try:
                f, sf = self.fit_gausses(p)
                r[i]["fit_s"] = f
                r[i]["sfit_s"] = sf
            except Exception:
                pass
            try:
                f, sf = self.fit_gausses(p, f = "dg")
                r[i]["fit_d"] = f
                r[i]["sfit_d"] = sf
            except Exception:
                pass
                
                
                
        return r




@export
@strax.takes_config(
    strax.Option(
        name='electron_drift_velocity',
        help='Vertical electron drift velocity in mm/ns',
        default_by_run=utils.GetDriftVelocity
    ),
)
class EventPositions(strax.Plugin):
    depends_on = ('event_basics',)
    dtype = [
        ('x', np.float32,
         'Interaction x-position (mm)'),
        ('y', np.float32,
         'Interaction y-position (mm)'),
        ('z', np.float32,
         'Interaction z-position (mm)'),
        ('r', np.float32,
         'Interaction r-position (mm)'),
        ('theta', np.float32,
         'Interaction angular position (radians)')]

    def compute(self, events):
        scale = 1
        z_obs = - self.config['electron_drift_velocity'] * events['drift_time']

        orig_pos = np.vstack([events['x_s2'], events['y_s2'], z_obs]).T
        r_obs = np.linalg.norm(orig_pos[:, :2], axis=1)

        result = dict(x=orig_pos[:, 0] * scale,
                      y=orig_pos[:, 1] * scale,
                      z=z_obs,
                      r=r_obs,
                      theta=np.arctan2(orig_pos[:, 1], orig_pos[:, 0]))

        return result



@strax.takes_config(
    strax.Option(
        'electron_lifetime',
        help="Electron lifetime (ns)",
        default_by_run=utils.GetELifetime)
)
class CorrectedAreas(strax.Plugin):
    depends_on = ['event_basics', 'event_positions']
    dtype = [('cs1', np.float32, 'Corrected S1 area (PE)'),
             ('cs2', np.float32, 'Corrected S2 area (PE)')]

    def setup(self):
        self.s1_map = InterpolatingMap(
            get_resource(self.config['s1_relative_lce_map']))
        self.s2_map = InterpolatingMap(
            get_resource(self.config['s2_relative_lce_map']))

    def compute(self, events):
        event_positions = np.vstack([events['x'], events['y'], events['z']]).T
        s2_positions = np.vstack([events['x_s2'], events['y_s2']]).T
        lifetime_corr = np.exp(
            events['drift_time'] / self.config['electron_lifetime'])

        return dict(
            cs1=events['s1_area'] / self.s1_map(event_positions),
            cs2=events['s2_area'] * lifetime_corr / self.s2_map(s2_positions))


@strax.takes_config(
    strax.Option('bottom_pmts', default=[0], help='PMTs in the bottom array'),
    strax.Option(
        'g1',
        help="S1 gain in PE / photons produced",
        default=0.1),
    strax.Option(
        'g2',
        help="S2 gain in PE / electrons produced",
        default=30),
    strax.Option(
        'lxe_w',
        help="LXe work function in quanta/eV",
        default=13.7e-3),
)
class EnergyEstimates(strax.Plugin):
    depends_on = ['corrected_areas']
    dtype = [
        ('e_light', np.float32, 'Energy in light signal (keV)'),
        ('e_charge', np.float32, 'Energy in charge signal (keV)'),
        ('e_ces', np.float32, 'Energy estimate (keV_ee)')]

    def compute(self, events):
        w = self.config['lxe_w']
        el = w * events['cs1'] / self.config['g1']
        ec = w * events['cs2'] / self.config['g2']
        return dict(e_light=el,
                    e_charge=ec,
                    e_ces=el+ec)


class EventInfo(strax.MergeOnlyPlugin):
    depends_on = ['events',
                  'event_basics', 'event_positions', 'corrected_areas',
                  'energy_estimates']
    save_when = strax.SaveWhen.ALWAYS
