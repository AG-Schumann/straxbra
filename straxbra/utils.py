from pymongo import MongoClient
from pymongo.son_manipulator import ObjectId
import os
import numpy as np
import datetime
import time


__client = MongoClient(os.environ['MONGO_DAQ_URI'])
db = __client['xebra_daq']
experiment = 'xebra'
MAX_RUN_ID = 999999  # because reasons

def _GetRundoc(run_id):
    query = {'run_id' : min(int(run_id), MAX_RUN_ID), 'experiment' : experiment}
    doc = db['runs'].find_one(query)
    #if doc is None:
    #    raise ValueError('No run with id %d' % run_id)
    return doc  # returns None if no doc

def GetRawPath(run_id):
    doc = _GetRundoc(run_id)
    return '/data/storage/strax/raw/live'
    if doc is not None:
        try:
            return doc['data']['raw']['location']
        except KeyError:
            pass
    return '/data/storage/strax/raw/unsorted/%s' % run_id

def GetReadoutThreads(run_id):
    doc = _GetRundoc(run_id)
    if doc is not None:
        try:
            return doc['config']['processing_threads']['charon_reader_0']
        except KeyError:
            pass
    return 2

def GetGains(run_id):
    doc = _GetRundoc(run_id)
    if doc is None:
        return np.ones(8)
    run_start = ObjectId.from_datetime(doc['start'])
    try:
        earlier_doc = list(db['pmt_gains'].find({'run' : {'$lte' : run_id}}).sort([('run', -1)]).limit(1))[0]
    except IndexError:
        return np.ones(8)
    try:
        later_doc = list(db['pmt_gains'].find({'run' : {'$gte' : run_id}}).sort([('_id', 1)]).limit(1))[0]
    except IndexError:
        return np.array(earlier_doc['adc_to_pe'])
    earlier_cal = int(str(earlier_doc['_id'])[:8], 16)
    later_cal = int(str(later_doc['_id'])[:8], 16)
    return np.array([np.interp(doc['start'].timestamp(),
                                [earlier_cal,later_cal],
                                [earlier_doc['adc_to_pe'][ch], later_doc['adc_to_pe'][ch]])
                        for ch in range(len(earlier_doc['adc_to_pe']))])

def GetELifetime(run_id):
    return 10e3 # 10 us

def GetRunStart(run_id):
    rundoc = _GetRundoc(run_id)
    if rundoc is not None:
        return int(rundoc['start'].timestamp()*1e9)
    return int(time.time()*1e9)

def GetNChan(run_id):
    rundoc = _GetRundoc(run_id)
    if rundoc is not None:
        try:
            board_id = rundoc['config']['boards'][0]['board']
            return len(rundoc['config']['channels'][str(board_id)])
        except KeyError:
            pass
    return 8


def GetDriftVelocity(run_id):
    drift_length = 7  # cm
    rundoc = _GetRundoc(run_id)
    if rundoc is not None:
        # from Jelle's thesis: v (mm/us) = 0.71*field**0.15 (V/cm)
        gate_mean =  rundoc['cathode_mean'] - 280 * rundoc['cathode_current_mean']
        return 7.1e-4*((rundoc['cathode_mean'] - gate_mean)/drift_length)**0.15
    return 1.8e-3  # 500 V/cm
