from pymongo import MongoClient
import os
import time
from datetime import datetime
import runs_todo_work

uri = "mongodb://xebra_daq:PApn6yIVQpbQSe39Nm8fAA==@192.168.131.2:27017"

_client = MongoClient(uri)
db = _client['xebra_daq']


bool_last = -1

while True:
    
    bool_now = str(runs_todo_work.bool_last_run_finished(db, verbose = False))
    if not bool_now == bool_last:
        bool_last = bool_now
        
        print("\n\33[33m" + str(datetime.now()) + ":\33[0m ", bool_now, flush = True, end = "")
        bool_tmp = str(runs_todo_work.bool_last_run_finished(db, verbose = True))
        
    else:
        print(".", end = "", flush = True)
    
    time.sleep(1)


print("\ndone\n")
