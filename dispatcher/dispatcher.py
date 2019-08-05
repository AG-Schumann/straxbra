from pymongo import MongoClient
import datetime
import time
import signal
import threading
from functools import partial
from subprocess import Popen, PIPE, TimeoutExpired
import os
import os.path as osp
import shutil
import zmq
import logging
import logging.handlers
import re
from blosc import decompress
import numpy as np


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
    def __init__(self, daq_db, logger, raw_dir):
        self.logger = logger
        self.db = daq_db
        self.root_raw_dir = raw_dir
        self.sh = SignalHandler()
        self.schedule = Scheduler(self.sh)
        self.schedule.start()
        self.status_map = ['idle','arming','armed','running','error','unknown']
        #self.context = zmq.Context()
        #self.led_sock = self.context.socket(zmq.REQ)
        #self.led_sock.connect("tcp://192.168.178.3:6000")
        self.stop_id = None
        self.current_run_id = None
        self.armed_for_id = None
        msg = ''
        try:
            self.Spatch()
        except Exception as e:
            msg = 'Caught a %s: %s' % (type(e), e)
        else:
            msg = ''
        finally:
            self.close(msg)
            self.logger.info('Closing with message: %s' % msg)

    def __del__(self):
        self.close()
        return

    def close(self, msg=''):
        self.schedule.join()
        self.SetStatus(active=False, status='offline', msg=msg)

    def GetNextRunId(self):
        abs_run_number = len(os.listdir(os.path.join(root_raw_dir, 'unsorted')))
        return '%05d' % (abs_run_number) # counts from 0 so no +1

    def InsertNewRundoc(self, mode, user, run_id_abs, config_override, comment):
        self.logger.info('Generating rundoc')
        rundoc = {
                'mode' : mode,
                'user' : user,
                'run_id_unsrt' : int(run_id_abs),
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
        if 'includes' in cfg_doc:
            for sub_cfg in cfg_doc['includes']:
                rundoc['config'].update(self.db['options'].find_one({'name' : sub_cfg}))
        rundoc['config'].update(cfg_doc)
        if config_override:
            rundoc['config'].update(config_override)
        del rundoc['config']['_id']
        rundoc['data'] = {'raw' :
                {'location': osp.join(rundoc['config']['strax_output_path'],run_id_abs)}}
        self.db['runs'].insert_one(rundoc)
        os.chdir(os.path.join(root_raw_dir, rundoc['experiment']))
        os.symlink(f'../unsorted/{run_id_abs}', run_id)
        # TODO this line fails???
        #unsrt_dir = raw_root_dir + '/unsorted'
        self.logger.info('Rundoc inserted, run id %s' % run_id)
        return run_id

    def EndRun(self, run_id):
        self.logger.debug('Ending run %s' % run_id)
        doc = self.db['runs'].find_one({'run_id_unsrt' : int(run_id)})
        if doc is None:
            self.logger.error('Invalid run error')
            self.SetStatus(msg='This error should not have happened, what did you do?')
            return
        updates = {}
        raw_dir = doc['data']['raw']['location']
        threads = doc['config']['processing_threads']['charon_reader_0']
        for _ in range(20):
            # it can take a bit for the daq to actually stop
            time.sleep(1)
            if 'THE_END' in os.listdir(raw_dir) and \
                    len(os.listdir(osp.join(raw_dir, 'THE_END'))) >= threads:
                break
        first_chunk = os.listdir(osp.join(raw_dir, '000000'))[0]
        with open(osp.join(raw_dir, '000000', first_chunk), 'rb') as f:
            rec = np.frombuffer(blosc.decompress(f.read()), dtype=record_dtype())[0]
            run_start = rec['time']
        # cleanup unnecessary folders
        for fn in os.listdir(raw_dir):
            if 'temp' in fn:
                shutil.rmtree(osp.join(raw_dir, fn))
        num_chunks = len(os.listdir(raw_dir))-1
        for chunk in list(range(num_chunks))[::-1]:
            if len(os.listdir(osp.join(raw_dir, f'{chunk:06d}'))) < threads:
                continue  # incomplete folder
            last_chunk = os.listdir(osp.join(raw_dir, f'{chunk:06d}'))[0]
            with open(osp.join(raw_dir, f'{chunk:06d}', last_chunk), 'rb') as f:
                rec = np.frombuffer(blosc.decompress(f.read()), dtype=record_dtype())[-1]
                duration = rec['time'] - run_start
                updates['end'] = doc['start'] + datetime.timedelta(seconds=duration/1e9)

        # figure out how much data we just made
        proc = Popen('du -chBM %s' % raw_dir, shell=True, stdout=PIPE, stderr=PIPE)
        try:
            out, err = proc.communicate(timeout=10)
        except TimeoutExpired:
            proc.kill()
            out, err = proc.communicate()
        m = re.search(b'(?P<size>[1-9][0-9]*(?:[,\\.][0-9]*[1-9])?)M\ttotal', out)
        if m:
            updates['data.raw.size'] = float(m.group('size').replace(b',',b'.'))

        updates['data.raw.location'] = os.path.join(root_raw_dir, doc['experiment'], '%05d' % doc['run_id'])
        self.db['runs'].update_one({'_id' : doc['_id']}, {'$set' : updates})
        return

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
        self.logger.info('Arming')  # TODO fix vv
        cmd_doc = {'host' : 'charon_reader_0', 'acknowledged' : [],
                'command' : '', 'user' : 'web'}
        experiment = self.db['options'].find_one({'name' : doc['mode']})['detector']
        run_id = self.GetNextRunId()
        cmd_doc['command'] = 'arm'
        cmd_doc['options_override'] = doc['config_override']
        cmd_doc['run_identifier'] = run_id
        self.armed_for_id = run_id
        cmd_doc['mode'] = doc['mode']
        self.db['options'].update_one({'name' : doc['mode']},
                            {'$set' : {'run_identifier' : run_id}})
        self.db['control'].insert_one(cmd_doc)
        self.SetStatus(msg='Arming for %s' % cmd_doc['mode'], goal='none')
        return

    def Start(self, doc):
        self.logger.info('Starting daq')  # TODO fix vv
        cmd_doc = {'host' : 'charon_reader_0', 'acknowledged' : [],
                'command' : '', 'user' : 'web'}
        cmd_doc['command'] = 'start'
        experiment = self.db['options'].find_one({'name' : doc['mode']})['detector']
        if self.armed_for_id is not None:
            run_id = self.armed_for_id
            self.armed_for_id = None
        else:
            run_id = self.GetNextRunId()
        self.logger.debug('Run id: %s' % run_id)
        cmd_doc['user'] = doc['user']
        cmd_doc['run_identifier'] = run_id
        self.current_run_id = run_id
        self.db['control'].insert_one(cmd_doc)
        run_id_rel = self.InsertNewRundoc(mode=doc['mode'], user=doc['user'], run_id_abs=run_id, config_override=doc['config_override'], comment=doc['comment'])
        self.SetStatus(msg='Run %s is live (%s)' % (run_id_rel, doc['mode']), run_id=run_id, goal='none', comment='')
        cmd_doc['command'] = 'stop'
        if '_id' in cmd_doc:
            del cmd_doc['_id']
        self.stop_id = self.schedule.Schedule(delay=doc['duration'], func = partial(
            self.Stop, doc))
        if experiment == 'xebra':  # FIXME
            self.logger.debug('Notifying strax-o-matic')
            if doc['mode'] not in ['led','noise']:
                targets = self.default_strax_targets
            else:
                targets = 'raw_records'
            self.db['system_control'].update_one({'subsystem' : 'straxinator'},
                    {'$set' : {'goal' : run_id_rel, 'targets' : targets}})
        else:
            self.logger.debug('Not straxing')
        return

    def Stop(self, doc):
        self.logger.info('Stopping daq')  # TODO fix vv
        cmd_doc = {'host' : 'charon_reader_0', 'acknowledged' : [],
                'command' : '', 'user' : 'web'}
        cmd_doc['command'] = 'stop'
        cmd_doc['user'] = doc['user']
        self.db['control'].insert_one(cmd_doc)
        if self.stop_id is not None:
            self.schedule.Unschedule(self.stop_id)
            self.stop_id = None
        self.SetStatus(status='online', goal='none', msg='')
        if self.current_run_id is not None:
            self.EndRun(run_id=self.current_run_id)
            self.current_run_id = None
        return

    def LED(self, doc):
        self.logger.debug('LED starting')
        led_cal_duration = 60*5
        self.SendToLed('arm')
        if 'config_override' not in doc:
            doc['config_override'] = {}
        doc['mode'] = 'led'
        self.Arm(doc)
        self.SetStatus(msg='Arming for LED calibration', goal='none')
        status = self.DAQStatus()
        while status != 'armed':
            time.sleep(1)
            if status not in ['idle','arming']:
                self.SetStatus(msg='LED arming failed')
                self.SendToLed('stop')
                return
            status = self.DAQStatus()
        self.SendToLed('start')
        doc['duration'] = led_cal_duration
        self.Start(doc)
        self.SetStatus(msg='Doing LED calibration')
        self.logger.debug('LED calibration starting')
        return

    def Spatch(self):
        self.SetStatus(active=True, status='online', msg='', goal='none')
        self.logger.info('Spatching')
        while self.sh.run:
            doc = self.db['system_control'].find_one({'subsystem' : 'daqspatcher'})
            daq_status = self.DAQStatus()
            goal = doc['goal']
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
    with MongoClient(os.environ['MONGO_DAQ_URI']) as client:
        raw_dir = '/data/storage/strax/raw'
        try:
            experiment = os.environ['EXPERIMENT_NAME']
            d = Dispatcher(client[f'{experiment}_daq'], logger, raw_dir)
        except Exception as e:
            logger.error(f'Caught a {type(e)}: {e}')
            try:
                d.sh.run = False
            except:
                pass
        else:
            pass

