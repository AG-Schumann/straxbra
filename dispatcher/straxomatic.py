import straxbra
import signal
import time
from pymongo import MongoClient
import os
import shutil
from prepare_folder import prepare_folder

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

    raw_path = '/data/storage/strax/raw/live/'
    
    
    
    bool_print = True
    while sh.run:
        
        if bool_print: 
            print("waiting for new data", flush = True)
        
        doc = collection.find_one({'subsystem' : 'straxinator'})
        if doc['goal'] == 'none':
            time.sleep(5)
            bool_print = False
            continue
        
        bool_print = True
        
        run_id = doc['goal']
        
        print("run_id " + str(run_id), flush = True)
        
        collection.update_one({'_id' : doc['_id']},
                {'$set' : {'status' : 'working',
                           'goal' : 'none', 'msg' : 'waiting for data to be written'}})
        int_slept = 0
        while len(os.listdir(raw_path)) < 2 :
            if os.path.isfile(raw_path + "DAQSPATCHER_OK"):
                print("DAQSPATCHER seems to be ready, no data was written to disk")
                break
            else:
                print("no data found yet, wating for 2 seconds ("+str(int_slept).rjust(3)+")", flush = True)
                time.sleep(2)
                int_slept += 1
            
        collection.update_one({'_id' : doc['_id']},
                {'$set' : {'status' : 'working',
                           'goal' : 'none', 'msg' : f'straxinating {run_id}'}})
        print("straxinating", flush = True)
        
        msg = straxinate(run_id, doc['targets'], max_workers=int(doc['max_workers']))
        
        print("done straxinating:", flush = True)
        
        if msg == '':
            # at least raw records successfully produced, delete reader data
            print("waiting till daqspatcher is ready", flush = True)
            collection.update_one({'_id' : doc['_id']},
                {'$set' : { 'msg' : 'waiting for dispatcher to be ready'}})
        
            while not os.path.isfile(raw_path + "DAQSPATCHER_OK"):
                time.sleep(2)
                
            
            print(prepare_folder(raw_path))
            
            
        updates = {'status' : 'idle', 'msg' : msg}
        collection.update_one({'_id' : doc['_id']}, {'$set' : updates})
        
            
        
    collection.update_one({'subsystem' : 'straxinator'},
            {'$set' : {'active' : False, 'status' : 'offline', 'msg' : ''}})
    return

if __name__ == '__main__':
    print("====================")
    print("starting straxinator")
    print("====================")
    
    with MongoClient(os.environ['MONGO_DAQ_URI']) as client:
        try:
            experiment = os.environ['EXPERIMENT_NAME']
            main(client[f'{experiment}_daq']['system_control'])
            print("  experiment: " + experiment)
        
        except Exception as e:
            print(f'Caught a {type(e)}: {e}')

