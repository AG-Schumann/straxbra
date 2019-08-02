from pymongo import MongoClient
from pymongo.son_manipulator import ObjectId
import os
import numpy as np
import datetime
import time


__client = MongoClient(os.environ['MONGO_DAQ_URI'])
try:
    experiment = os.environ['EXPERIMENT_NAME']
except KeyError:
    raise ImportError('No experiment name specified!')
db = __client[f'{experiment}_daq']
MAX_RUN_ID = 999999  # because reasons

def _GetRundoc(run_id):
    query = {'run_id' : min(int(run_id), MAX_RUN_ID), 'experiment' : experiment}
    doc = db['runs'].find_one(query)
    return doc  # returns None if no doc

def GetRawPath(run_id):
    doc = _GetRundoc(run_id)
    try:
        return doc['data']['raw']['location']
    except KeyError, TypeError:
        return '/data/storage/strax/raw/unsorted/%s' % run_id

def GetReadoutThreads(run_id):
    doc = _GetRundoc(run_id)
    try:
        return sum(doc['config']['processing_threads'].values())
    except KeyError, TypeError:
        return 1

def GetGains(run_id):
    doc = _GetRundoc(run_id)
    if doc is None:
        return np.ones(8)
    run_start = ObjectId.from_datetime(doc['start'])
    try:
        earlier_doc = list(db['pmt_gains'].find({'_id' : {'$lte' : run_start}}).sort([('_id', -1)]).limit(1))[0]
    except IndexError:
        return np.ones(8)
    try:
        later_doc = list(db['pmt_gains'].find({'_id' : {'$gte' : run_start}}).sort([('_id', 1)]).limit(1))[0]
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
    try:
        return int(rundoc['start'].timestamp()*1e9)
    except TypeError:
        return int(time.time()*1e9)

def GetNChan(run_id):
    rundoc = _GetRundoc(run_id)
    channels = 0
    try:
        cfg = rundoc['config']
        for board_cfg in cfg['boards']:
            channels += len(cfg['channels'][str(board_cfg['board'])])
        return channels
    except KeyError, TypeError:
        return 2  # strax has problems with 1 channel

