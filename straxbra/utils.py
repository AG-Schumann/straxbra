from pymongo import MongoClient
from pymongo.son_manipulator import ObjectId
import os
import numpy as np
import datetime
import time


__client = MongoClient(os.environ['DISPATCHER_URI'])
db = __client['xebra_daq']

def gaus(x, mean, sigma, amp):
    log_amp = np.log(amp) - 0.5*np.log(2*np.pi*sigma**2)
    log_exp = -(x - mu)**2/(2*sigma**2)
    return np.exp(log_amp + log_exp)

def GetRawPath(run_id):
    doc = db['runs'].find_one({'run_id' : int(run_id)})
    if doc is not None:
        return doc['data']['raw']['location']
    return '/data/storage/strax/raw/%s' % run_id

def GetReadoutThreads(run_id):
    doc = db['runs'].find_one({'run_id' : int(run_id)})
    if doc is not None:
        return doc['config']['processing_threads']['charon_reader_0']
    return 2

def GetGains(run_id):
    return np.ones(8)*1
    doc = db['runs'].find_one({'run_id' : int(run_id)})
    run_start = (str(doc['_id'])[:8] + '0'*len(str(doc['_id'])-8)).encode()
    earlier_doc = list(db['pmt_gains'].find({'_id' : {'$lte' : ObjectId(run_start)}}).sort([('_id', -1)]).limit(1))[0]
    later_doc = list(db['pmt_gains'].find({'_id' : {'$gte' : ObjectId(run_start)}}).sort([('_id', 1)]).limit(1))[0]
    if not later_doc:
        return ealier_doc['gains']
    return np.array([np.interp(int(run_start[:8],16),
                                [int(str(earlier_doc['_id'])[:8],16),
                                 int(str(later_doc['_id'])[:8],16)],
                                [earlier_doc['gains'][ch], later_doc['gains'][ch]])
                        for ch in range(len(earlier_doc['gains']))])

def GetELifetime(run_id):
    return 10e3 # 10 us

def UpdateGains(bin_centers, histos, fit_results, fit_uncertainties):
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
    rundoc = db['runs'].find_one({'run_id' : int(run_id)})
    if rundoc is not None:
        return int(rundoc.timestamp()*1e9)
    return int(time.time()*1e9)

def GetNChan(run_id):
    rundoc = db['runs'].find_one({'run_id' : int(run_id)})
    if rundoc is not None:
        board_id = rundoc['config']['boards'][0]['board']
        return len(rundoc['config']['channels'][str(board_id)])
    return 8
