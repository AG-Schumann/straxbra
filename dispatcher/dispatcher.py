from pymongo import MongoClient
import datetime
import time
import signal
import threading
from functools import partial
from subprocess import Popen, PIPE, TimeoutExpired
import os
import os.path as osp
#import zmq
import logging
import logging.handlers
import re
from blosc import decompress
import numpy as np
from strax import record_dtype
import shutil
from prepare_folder import prepare_folder
import runs_todo_work
import requests
import json


class Scheduler(threading.Thread):
    def __init__(self, sh, logger):
        threading.Thread.__init__(self)
        self.sh = sh
        self.logger = logger
        self.queue_lock = threading.RLock()
        self.queue = []
        self.event_id = 0

    def run(self):
        default_sleep_time = 10
        logging.info('Scheduler starting')
        while self.sh.run:
            with self.queue_lock:
                if len(self.queue) > 0:
                    next_event_time = self.queue[0]['delay']
                    diff = min(next_event_time, default_sleep_time)
                    for i in range(len(self.queue)):
                        self.queue[i]['delay'] -= diff
                else:
                    diff = default_sleep_time
            time.sleep(diff)
            with self.queue_lock:
                if len(self.queue) > 0 and self.queue[0]['delay'] < 0.01:
                    self.queue.pop(0)['func']()
        logger.info('Scheduler ending')
        logger.debug('%i jobs scheduled' % self.event_id)
        if len(self.queue):
            logger.debug('%i jobs un-completed' % len(self.queue))
        return

    def Schedule(self, delay, func):
        self.logger.debug('Scheduling "%s" with %s delay (id %i)' % (
            str(func).split(' ')[1], delay, self.event_id))
        with self.queue_lock:
            self.queue.append({'delay' : delay, 'func' : func, 'id' : self.event_id})
            self.queue.sort(key=lambda x : x['delay'])
            last_event_id = self.event_id
            self.event_id += 1
        return last_event_id

    def Unschedule(self, id_to_remove):
        with self.queue_lock:
            for i in range(len(self.queue)):
                if self.queue[i]['id'] == id_to_remove:
                    _ = self.queue.pop(i)
                    break
        return


class SignalHandler(object):
    def __init__(self, logger):
        self.run = True
        signal.signal(signal.SIGINT, self.interrupt)
        signal.signal(signal.SIGTERM, self.interrupt)
        self.logger = logger

    def interrupt(self, *args):
        self.logger.info(f'Caught signal {args[0]}')
        self.run = False


class Dispatcher(object):
    def __init__(self, client, logger, raw_dir):
        self.logger = logger
        self.db = client['xebra_daq']
        self.slow_db = client['xebra_data']
        self.root_raw_dir = raw_dir
        self.raw_dir = str(raw_dir) + "/live"
        self.sh = SignalHandler(self.logger)
        self.schedule = Scheduler(self.sh, self.logger)
        self.schedule.start()
        self.status_map = ['idle','arming','armed','running','error','unknown']
        #self.context = zmq.Context()
        #self.led_sock = self.context.socket(zmq.REQ)
        #self.led_sock.connect("tcp://192.168.178.3:6000")
        self.stop_id = None
        self.current_run_id = None
        self.armed_for_id = None
        self.default_strax_targets = 'event_positions'
        
        
    def __del__(self):
        self.close()
        return

    def close(self, msg=''):
        self.sh.run = False
        self.schedule.join()
        self.SetStatus(active=False, status='offline', msg=msg)
        self.logger.info(msg)

    def InsertNewRundoc(self, mode, user, config_override, comment):
        self.logger.info('Generating rundoc')
        rundoc = {
                'mode' : mode,
                'user' : user,
                'config' : {},
                'start' : datetime.datetime.utcnow(),
                'comment' : comment,
                }
        cfg_doc = self.db['options'].find_one({'name' : mode})
        rundoc['experiment'] = cfg_doc['detector']
        run_id = 0
        for row in self.db['runs'].find({'experiment' : cfg_doc['detector']}).sort([('run_id', -1)]).limit(1):
            run_id = row['run_id']+1
        rundoc['run_id'] = run_id
        run_id = f'{run_id:05d}'
        self.logger.debug('Assigned run id %s' % run_id)
        if 'includes' in cfg_doc:
            self.logger.debug('Adding config includes')
            for sub_cfg in cfg_doc['includes']:
                rundoc['config'].update(self.db['options'].find_one({'name' : sub_cfg}))
        rundoc['config'].update(cfg_doc)
        if config_override:
            rundoc['config'].update(config_override)
        del rundoc['config']['_id']
        self.logger.debug('Inserting rundoc')
        self.db['runs'].insert_one(rundoc)
        self.logger.info('Rundoc inserted, run id %s' % run_id)
        return run_id

    def EndRun(self, run_id):
        self.logger.debug('Ending active run')
        
         
        doc = self.db['runs'].find_one({'end' : {'$exists' : 0}})
        # doc = self.db['runs'].find({'run_id' : {'$exists' : 1}, 'end' : {'$exists' : 0}}).sort({'run_id': 1}).limit(1)
        
        
        
        
        self.logger.debug('maybe found a document. Run_id: ' + str(doc["run_id"]))
        
        if doc is None:
            self.logger.error('Invalid run error')
            self.SetStatus(msg='This error should not have happened, what did you do?')
            return
        updates = {}
        
        # check if thread amount exists
        threads = doc['config']['processing_threads']['charon_reader_0']
        self.logger.debug('threads found in config and set to ' + str(threads))
        
        
        self.logger.debug('Waiting for daq to stop')
        
        for _ in range(20):
            # it can take a bit for the daq to actually stop
            self.logger.debug("sleeping: " + str(_))
            time.sleep(1)
            
            count_folders = len(os.listdir(self.raw_dir))
            
            if requests.get("http://localhost/control/get_status").json()["daqstatus"]:
                self.logger.debug("daq is now idle")
                break
            
            if 'THE_END' in os.listdir(self.raw_dir) and \
                    len(os.listdir(osp.join(self.raw_dir, 'THE_END'))) >= threads:
                self.logger.debug("THE_END found")
                break
                
        self.logger.debug(str(count_folders) + " folders in folder)")
        
        if(count_folders > 0):
            self.logger.debug('Daq stopped')
            first_chunk = os.listdir(osp.join(self.raw_dir, '000000'))[0]
            with open(osp.join(self.raw_dir, '000000', first_chunk), 'rb') as f:
                rec = np.frombuffer(decompress(f.read()), dtype=record_dtype())[0]
                run_start = rec['time']
            # cleanup unnecessary folders
            for fn in os.listdir(self.raw_dir):
                if 'temp' in fn:
                    shutil.rmtree(osp.join(self.raw_dir, fn))
            chunks = sorted(os.listdir(self.raw_dir))
            for chunk in chunks[::-1]:
                if len(os.listdir(osp.join(self.raw_dir, chunk))) < threads:
                    continue  # incomplete folder
                last_chunk = os.listdir(osp.join(self.raw_dir, chunk))[0]
                try:
                    with open(osp.join(self.raw_dir, chunk, last_chunk), 'rb') as f:
                        rec = np.frombuffer(decompress(f.read()), dtype=record_dtype())[-1]
                        duration = rec['time'] - run_start
                        updates['end'] = doc['start'] + datetime.timedelta(seconds=duration/1e9)
                except Exception:
                    continue  # mark for removal?
                break
        else:
            self.logger.debug("failed finding last chunck, using current time as end")
            updates['end'] = datetime.datetime.utcnow()



        if doc['mode'] not in ['led', 'noise']:
            self.logger.debug('Getting SC data')
            try:
                updates.update(self.GetMeshVoltages(doc['start'], updates['end']))
            except:
                self.logger.debug('failed')
                
        self.logger.debug('Updating rundoc')
        self.db['runs'].update_one({'_id' : doc['_id']}, {'$set' : updates})
        self.logger.debug('waiting two seconds to continue')
        
        self.logger.debug('waiting for straxinator to be ready')
        
        f = open(self.raw_dir + "/DAQSPATCHER_OK", "x")
        f.write("OK")
        f.close
        
        self.SetStatus(msg='waiting for straxinator to finish')
        int_straxinator_counter = 0
        while True:
            stat_straxinator = self.db["system_control"].find_one({"subsystem": "straxinator"})["status"]
            if stat_straxinator == "idle":
                break
            
            self.logger.debug("straxinator not ready yet ("+stat_straxinator+"|" + str(int_straxinator_counter) + ")")
            int_straxinator_counter += 1
            time.sleep(2)
        
        self.SetStatus(msg='')
        
        self.logger.debug('Run ended')
        
        
        
        return

    def GetMeshVoltages(self, run_start, run_end):
        coll = self.slow_db['caen_n1470']
        pipeline = [
                {'$match' : {'_id' : {'$gte' : ObjectId.from_datetime(run_start),
                                      '$lte' : ObjectId.from_datetime(run_end)}}},
                {'$group' : {'_id' : None,
                             'cathode_mean' : {'$avg' : '$cathode_voltage'},
                             'anode_mean' : {'$avg' : '$anode_voltage'},
                             'cathode_dev' : {'$stdDevPop' : '$cathode_voltage'},
                             'anode_dev' : {'$stdDevPop' : '$anode_voltage'}}},
                {'$project' : {'_id' : 0, 'cathode_mean' : 1, 'anode_mean' : 1,
                                'cathode_dev' : 1, 'anode_dev' : 1}},
                ]
        return coll.aggregate(pipeline)

    def DAQStatus(self):
        for row in self.db['status'].find({}).sort([('_id', -1)]).limit(1):
            if time.time()-int(str(row['_id'])[:8],16) > 20:
                return 'offline'
            return self.status_map[row['status']]

    def SetStatus(self, **kwargs):
        self.db['system_control'].update_one({'subsystem' : 'daqspatcher'},
                {'$set' : kwargs})
        return

    def SendToLed(self, msg):
        #self.led_sock.send(msg)
        #ret = self.led_sock.recv()
        return

    def Arm(self, doc):
        self.logger.info('Preparing folder')
        prepare_folder(self.raw_dir)
        
        self.logger.info('Arming for %s' % doc['mode'])
        cmd_doc = {'host' : ['charon_reader_0'], 'acknowledged' : [],
                'command' : '', 'user' : doc['user']}
        experiment = self.db['options'].find_one({'name' : doc['mode']})['detector']
        cmd_doc['command'] = 'arm'
        cmd_doc['options_override'] = doc['config_override']
        cmd_doc['run_identifier'] = 'live'
        cmd_doc['mode'] = doc['mode']
        self.db['options'].update_one(
                {'name' : doc['mode']},
                {'$set' : {'run_identifier' : 'live'}}
        )
        self.logger.debug('Inserting command doc')
        self.db['control'].insert_one(cmd_doc)
        self.SetStatus(msg='Arming for %s' % cmd_doc['mode'], goal='none')
        
        self.logger.debug('Done arming')
        
        return

    def Start(self, doc):
        if os.path.isfile(self.raw_dir + "/DAQSPATCHER_OK"):
            os.remove(self.raw_dir + "/DAQSPATCHER_OK")
            self.logger.info('removed ready to end file')
            
        
        
        self.logger.info('Starting daq')
        cmd_doc = {'host' : ['charon_reader_0'], 'acknowledged' : [],
                'command' : '', 'user' : doc['user']}
        cmd_doc['command'] = 'start'
        experiment = self.db['options'].find_one({'name' : doc['mode']})['detector']
        cmd_doc['run_identifier'] = 'live'
        self.logger.debug('Issuing command')
        self.db['control'].insert_one(cmd_doc)
        if '_id' in cmd_doc:
            del cmd_doc['_id']
        self.logger.debug('Scheduling stop')
        cmd_doc['command'] = 'stop'
        
        self.logger.debug('Duration: ' + str(doc['duration']))
        
        self.stop_id = self.schedule.Schedule(delay=doc['duration'], func = partial(
            self.Stop, doc))
        self.logger.debug('Adding rundoc')
        run_id_rel = self.InsertNewRundoc(mode=doc['mode'], user=doc['user'],
                config_override=doc['config_override'],
                comment=doc['comment'])
        self.current_run_id = run_id_rel
        self.SetStatus(msg='Run %s is live (%s)' % (run_id_rel, doc['mode']),
                run_id=run_id_rel, goal='none', comment='')
        if experiment == 'xebra':  # FIXME
            self.logger.debug('Notifying strax-o-matic')
            if doc['mode'] not in ['led','noise']:
                targets = self.default_strax_targets
            else:
                targets = 'raw_records'
            self.db['system_control'].update_one({'subsystem' : 'straxinator'},
                    {'$set' : {'goal':run_id_rel, 'targets':targets, 'duration': doc['duration']}})
        else:
            self.logger.debug('Not straxing')
        return

    def Stop(self, doc):
        self.logger.info('Stopping daq')
        cmd_doc = {'host' : ['charon_reader_0'], 'acknowledged' : [],
                'command' : '', 'user' : doc['user']}
        cmd_doc['command'] = 'stop'
        cmd_doc['user'] = doc['user']
        self.db['control'].insert_one(cmd_doc)
        if self.stop_id is not None:
            self.logger.debug('Unscheduling stop command')
            self.schedule.Unschedule(self.stop_id)
            self.stop_id = None
        self.SetStatus(status='online', goal='none', msg='')
        
        self.logger.debug('Ending run')
        self.logger.debug('test0-0')
        self.EndRun(run_id=self.current_run_id)
        self.logger.debug('test0-1')
        self.current_run_id = None
        self.logger.info(" waiting for data")
        return

    def LED(self, doc):
        self.logger.debug('LED starting')
        #led_cal_duration = 60*3
        led_cal_duration = 60*1
        # led_cal_duration = 30
        
        self.SendToLed('arm')
        if 'config_override' not in doc:
            doc['config_override'] = {}
        doc['mode'] = 'led'
        self.Arm(doc)
        self.SetStatus(msg='Arming for LED calibration', goal='none')
        self.logger.debug('Waiting for daq to arm')
        
        self.db['system_control'].update_one({'subsystem':'daqspatcher'}, {'$set':{"duration":led_cal_duration}})
        
        status = self.DAQStatus()
        while status != 'armed':
            time.sleep(1)
            if status not in ['idle','arming']:
                self.SetStatus(msg='LED arming failed')
                self.SendToLed('stop')
                return
            status = self.DAQStatus()
        self.logger.debug('Daq armed')
        self.SendToLed('start')
        doc['duration'] = led_cal_duration
        self.Start(doc)
        self.SetStatus(msg='Doing LED calibration')
        self.logger.debug('LED calibration starting')
        return

    def Spatch(self):
        bool_print = True
        self.SetStatus(active=True, status='online', msg='', goal='none')
        self.logger.info('Spatching')
        while self.sh.run:
            
            if(runs_todo_work.bool_last_run_finished(self.db)):
                stat_straxinator = self.db["system_control"].find_one({"subsystem": "straxinator"})["status"]
                if not stat_straxinator == "idle":
                    self.logger.info('found ready to copy run from runs_todo, but straxinator is not ready')
                    
                else:
                    self.logger.info('found ready to copy run from runs_todo, waiting 2 seconds')
                    self.logger.info('  copying ...')
                    
                    threading.Thread(
                        target = runs_todo_work.copy_next_run,
                        args=(
                            self.db,
                            self.logger
                        )
                    ).start()
                    
                    self.logger.info('  thread started?')
                    
                
                
            doc = self.db['system_control'].find_one({'subsystem' : 'daqspatcher'})
            daq_status = self.DAQStatus()
            goal = doc['goal']
            if (not goal == "none") or bool_print:
                bool_print = False
                self.logger.info('  status: ' + str(daq_status))
                self.logger.info('  goal:   ' + str(goal))
                self.logger.info(" waiting for data")
                
            if daq_status == 'offline':
                time.sleep(5)
                continue
            if goal == 'arm':
                if daq_status == 'idle':
                    self.Arm(doc)
                else:
                    self.SetStatus(msg='Can\'t arm, daq is %s not idle' % daq_status,
                            goal='none')
            elif goal == 'start':
                if daq_status == 'armed':
                    self.Start(doc)
                else:
                    self.SetStatus(msg='Can\'t start, daq is %s not armed' % daq_status,
                        goal='none')
            elif goal == 'stop':
                if daq_status in ['armed', 'running']:
                    self.logger.info("TEST")
                    bool_print = True
                    self.Stop(doc)
                else:
                    self.SetStatus(msg=('Can\'t stop, daq is %s not '
                        'armed/running' % daq_status), goal='none')
            elif goal == 'led':
                if daq_status == 'idle':
                    self.LED(doc)
                else:
                    self.SetStatus(msg=('Can\'t do LED calibration, daq is %s not '
                            'idle' % daq_status), goal='none')
        
            time.sleep(1)
            # end of while loop

        self.logger.info('Daqspatcher returning')
        return

if __name__ == '__main__':
    logger = logging.getLogger('dispatcher')
    h = logging.handlers.TimedRotatingFileHandler(
            osp.join('/data/storage/logs/dispatcher',
                datetime.date.today().isoformat() + '.log'),
            when='midnight', delay=True)
    h.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
    logger.addHandler(h)
    logger.setLevel(logging.DEBUG)
    with MongoClient(os.environ['MONGO_DAQ_URI']) as client:
        raw_dir = '/data/storage/strax/raw'
        try:
            experiment = os.environ['EXPERIMENT_NAME']
            d = Dispatcher(client, logger, raw_dir)
            d.Spatch()
        except Exception as e:
            msg = 'Caught a %s: %s' % (type(e), e)
        else:
            msg = ''
        finally:
            d.close(msg)
