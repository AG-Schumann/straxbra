from pymongo import MongoClient
from pymongo.son_manipulator import ObjectId
import os
import numpy as np
import datetime
import time

n_pmts = 8
drift_length = 7  # in cm
MAX_RUN_ID = 999999  # because reasons

class RunsDBInterface:

    def __init__(self, experiment):
        __client = MongoClient(os.environ['MONGO_DAQ_URI'])
        self.db = __client['xebra_daq']
        self.experiment = experiment

    def _GetRundoc(self, run_id):
        query = {
            'run_id' : min(int(run_id), MAX_RUN_ID),
            'experiment' : self.experiment
        }
        doc = self.db['runs'].find_one(query)
        #if doc is None:
        #    raise ValueError('No run with id %d' % run_id)
        return doc  # returns None if no doc

    def GetRawPath(self, run_id):
        doc = self._GetRundoc(run_id)
        return '/data/storage/strax/raw/live'
        if doc is not None:
            try:
                return doc['data']['raw']['location']
            except KeyError:
                pass
        return '/data/storage/strax/raw/unsorted/%s' % run_id

    def GetReadoutThreads(self, run_id):
        doc = self._GetRundoc(run_id)
        if doc is not None:
            try:
                return doc['config']['processing_threads']['charon_reader_0']
            except KeyError:
                pass
        return 2

    def GetGains(self, run_id):
        doc = self._GetRundoc(run_id)

        if doc is None:
            return np.ones(n_pmts)
        run_start = datetime.datetime.timestamp(doc['start'])
        try:
            earlier_doc = list(self.db['pmt_gains'].find({'time' : {'$lte' : run_start}}).sort([('time', -1)]).limit(1))[0]
        except IndexError:
            return np.ones(n_pmts)
        try:
            later_doc = list(self.db['pmt_gains'].find({'time' : {'$gte' : run_start}}).sort([('time', 1)]).limit(1))[0]
        except IndexError:
            return np.array(earlier_doc['adc_to_pe'])
        #earlier_cal = int(str(earlier_doc['_id'])[:8], 16)
        #later_cal = int(str(later_doc['_id'])[:8], 16)
        earlier_cal = earlier_doc['time']
        later_cal = later_doc['time']
        return np.array([np.interp(doc['start'].timestamp(),
                                    [earlier_cal,later_cal],
                                    [earlier_doc['adc_to_pe'][ch], later_doc['adc_to_pe'][ch]])
                            for ch in range(len(earlier_doc['adc_to_pe']))])

    def GetELifetime(self, run_id):
        return 10e3 # 10 us

    def GetRunStart(self, run_id):
        rundoc = self._GetRundoc(run_id)
        if rundoc is not None:
            return int(rundoc['start'].timestamp()*1e9)
        return int(time.time()*1e9)

    def GetNChan(self, run_id):
        rundoc = self._GetRundoc(run_id)
        if rundoc is not None:
            try:
                board_id = rundoc['config']['boards'][0]['board']
                return len(rundoc['config']['channels'][str(board_id)])
            except KeyError:
                pass

        return n_pmts


    def GetDriftVelocity(self, run_id):
        rundoc = self._GetRundoc(run_id)
        if rundoc is not None:
            if 'cathode_mean' in rundoc:
                # from Jelle's thesis: v (mm/us) = 0.71*field**0.15 (V/cm)
                gate_mean =  rundoc['cathode_mean'] - 280 * rundoc['cathode_current_mean']
                return 7.1e-4*((rundoc['cathode_mean'] - gate_mean)/drift_length)**0.15
        return 1.8e-3  # 500 V/cm
