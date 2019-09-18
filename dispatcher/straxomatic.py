import straxbra
import signal
import time
from pymongo import MongoClient
import os
import shutil


class SignalHandler:
    def __init__(self):
        self.run = True
        signal.signal(signal.SIGINT, self.interrupt)
        signal.signal(signal.SIGTERM, self.interrupt)

    def interrupt(self, *args):
        print(' Received interrupt signal %i' % args[0])
        self.run = False

def straxinate(run_id, targets, max_workers):
    try:
        straxbra.XebraContext().make(run_id, targets, max_workers=max_workers)
    except Exception as e:
        try:
            straxbra.XebraContext().make(run_id, 'raw_records', max_workers=max_workers)
        except Exception as ex:
            return f'Strax threw a {type(ex)}: {ex}'
        return ''
    return ''

def main(collection):
    sh = SignalHandler()
    collection.update_one({'subsystem' : 'straxinator'},
            {'$set' : {'active' : True, 'status' : 'idle', 'msg' : ''}})
    while sh.run:
        doc = collection.find_one({'subsystem' : 'straxinator'})
        if doc['goal'] == 'none':
            time.sleep(5)
            continue
        run_id = doc['goal']
        collection.update_one({'_id' : doc['_id']},
                {'$set' : {'status' : 'working',
                           'goal' : 'none', 'msg' : f'straxinating {run_id}'}})
        msg = straxinate(run_id, doc['targets'], max_workers=int(doc['max_workers']))
        updates = {'status' : 'idle', 'msg' : msg}
        collection.update_one({'_id' : doc['_id']}, {'$set' : updates})
        if msg == '':
            # at least raw records successfully produced, delete reader data
            raw_path = '/data/storage/strax/raw/live'
            shutil.rmtree(raw_path)

    collection.update_one({'subsystem' : 'straxinator'},
            {'$set' : {'active' : False, 'status' : 'offline', 'msg' : ''}})
    return

if __name__ == '__main__':
    with MongoClient(os.environ['MONGO_DAQ_URI']) as client:
        try:
            experiment = os.environ['EXPERIMENT_NAME']
            main(client[f'{experiment}_daq']['system_control'])
        except Exception as e:
            print(f'Caught a {type(e)}: {e}')

