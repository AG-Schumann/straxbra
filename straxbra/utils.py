from pymongo import MongoClient
from pymongo.son_manipulator import ObjectId
import os
import numpy as np
import datetime
import time


__client = MongoClient(os.environ['DISPATCHER_URI'])
db = __client['xebra_daq']
experiment = 'xebra'

def _GetRundoc(run_id):
    query = {'run_id' : int(run_id), 'experiment' : experiment}
    doc = db['runs'].find_one(query)
    if doc is None:
        raise ValueError('No run with id %d' % run_id)
    return doc

def gaus(x, mean, sigma, amp):
    log_amp = np.log(amp) - 0.5*np.log(2*np.pi*sigma**2)
    log_exp = -(x - mu)**2/(2*sigma**2)
    return np.exp(log_amp + log_exp)

def GetRawPath(run_id):
    doc = _GetRundoc(run_id)
    if doc is not None:
        return doc['data']['raw']['location']
    return '/data/storage/strax/raw/unsorted/%s' % run_id

def GetReadoutThreads(run_id):
    doc = _GetRundoc(run_id)
    if doc is not None:
        return doc['config']['processing_threads']['charon_reader_0']
    return 2

def GetGains(run_id):
    doc = _GetRundoc(run_id)
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

def UpdateGains(bin_centers, histos, fit_results, fit_uncertainties):
    doc = {}
    db['pmt_gains'].insert_one({
            'bin_centers' : bin_centers.tolist(),
            'histograms' : histos.tolist(),
            'gains' : fit_results[:,3].tolist(),
            'fit_results' : fit_results.tolist(),
            'fit_uncertainties' : fit_uncertainties.tolist(),
    })

def GetLastGains():
    doc = list(db['pmt_gains'].find({}).sort([('_id', -1)]).limit(1))[0]
    if 'fit_results' in doc:
        return doc['fit_results']
    return None

def GetRunStart(run_id):
    rundoc = _GetRundoc(run_id)
    if rundoc is not None:
        return int(rundoc['start'].timestamp()*1e9)
    return int(time.time()*1e9)

def GetNChan(run_id):
    rundoc = _GetRundoc(run_id)
    if rundoc is not None:
        print(rundoc)
        try:
            board_id = rundoc['config']['boards'][0]['board']
            return len(rundoc['config']['channels'][str(board_id)])
        except KeyError:
            return 8
    return 8

def RemoveRaw(raw_path):
    run_id = int(raw_path.split('/')[-1])
    db['runs'].update_one({'run_id' : run_id, 'experiment' : experiment},
            {'$set' : {'data.raw.location' : 'deleted'}})
