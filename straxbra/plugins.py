import os
import shutil

import numpy as np
from scipy.optimize import minimize
import keras

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
    depends_on = ('records',)
    data_kind = 'peaks'
    parallel = True
    rechunk_on_save = True
    

    def infer_dtype(self):
        return strax.peak_dtype(n_channels=self.config['n_channels'])

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
                                 max_duration=self.config['peak_max_duration'])
        strax.sum_waveform(peaks, r, adc_to_pe=self.config['to_pe'])
        peaks = peaks[peaks['dt'] > 0]  # removes strange edge case
        peaks = strax.split_peaks(peaks, r, self.config['to_pe'],
                                  min_height=self.config['split_min_height'],
                                  min_ratio=self.config['split_min_ratio'])

        strax.compute_widths(peaks)

        return peaks


@export
@strax.takes_config(
        strax.Option('top_pmts', track=False, default=list(range(1,7+1)),
                     type=list, help="Which PMTs are in the top array"),
        strax.Option('to_pe', track=False,
                     default_by_run=utils.GetGains,
                     help='PMT gains'),
)
class PeakBasics(strax.Plugin):
    """
    Stolen from straxen, extended with risetime. Also replaces
    aft for nonphysical peaks with nan.
    """
    __version__ = "0.0.1"
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
        r['max_pmt'] = np.argmax(p['area_per_channel'], axis=1)
        r['max_pmt_area'] = np.max(p['area_per_channel'], axis=1)

        area_top = (p['area_per_channel'][:, self.config['top_pmts']]
                    * self.config['to_pe'][self.config['top_pmts']].reshape(1, -1)).sum(axis=1)
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
                 default=100)
)
class PeakPositions(strax.Plugin):
    '''
    Position Reconstruction for XeBRA

    Version 0.0.1: Weighted Sum
    Version 0.0.2: LRF
    Version 0.0.3: Neural Network

    Status: September 2019, Version 0.0.3

    Position reconstruction for XeBRA using a Deep Feed Forward (DFF) Neural Network 
    with Keras trained on Geant4 MC simulations.
    '''
    __version__ = "0.0.3"
    dtype = [('x', np.float32,
              'Reconstructed S2 X position (mm), uncorrected'),
             ('y', np.float32,
              'Reconstructed S2 Y position (mm), uncorrected')]
    depends_on = ('peaks',)
    parallel = True

    def setup(self):
        ## PMT mask - select top PMTs
        self.pmt_mask = np.zeros_like(self.config['to_pe'], dtype=np.bool)
        self.pmt_mask[self.config['top_pmts']] = np.ones_like(self.pmt_mask[self.config['top_pmts']])
        ## Load the trained model from corresponding HDF5 file
        self.model_NN = keras.models.load_model('/data/workspace/nn_models/XeBRA_Position_Reconstruction_NN_Model_DualPhase_7TopPMTs.h5')
        
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
    __version__ = '0.0.1'
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
                         'range_50p_area', 'n_competing']:
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
class EventHermBasics(strax.LoopPlugin):
    __version__ = '0.0.1'
    depends_on = ('events',
                  'peak_basics', 'peak_classification',
                  'n_competing')

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
                      ((f'Main S{i} number of competing peaks',
                        f's{i}_n_competing'), np.int32)]
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
                         'range_50p_area', 'n_competing']:
                result[f's{s_i}_{prop}'] = s[prop]
            
        # Compute a drift time only if we have a valid S1-S2 pairs
        if len(main_s) == 2:
            result['drift_time'] = main_s[2]['time'] - main_s[1]['time']

        return result



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



@export
@strax.takes_config(
    strax.Option(
        name='electron_drift_velocity',
        help='Vertical electron drift velocity in mm/ns',
        default_by_run=utils.GetDriftVelocity
    ),
)
class EventHermPositions(strax.Plugin):
    depends_on = ('event_herm_basics',)
    dtype = [
        ('z', np.float32,
         'Interaction z-position (mm)')
        ]

    def compute(self, events):
        z_obs = - self.config['electron_drift_velocity'] * events['drift_time']
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
