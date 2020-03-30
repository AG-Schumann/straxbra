import os
import shutil

def prepare_folder(raw_path):
    int_deletetrys = 0
    while True:
        if os.path.exists(raw_path):
            if len(os.listdir(raw_path)) > 0:
                if int_deletetrys < 10:
                    int_deletetrys+=1
                    
                    shutil.rmtree(raw_path, ignore_errors = True)
                else:
                    return("maximum number of delete tries reached: " + str(int_deletetrys))
                    break
            else:
                return("empty folder for next run achieved in " + str(int_deletetrys) + " tries")
                break
        else:
            os.mkdir(raw_path)
            return("created folder for next run in " + str(int_deletetrys) + " tries")
            break




def bool_last_round_finished(db):
        
    json_last_run = list(db["runs"].aggregate(
        [
            {'$sort' : {"_id" : -1}},
            {'$limit' : 1}
        ]
    ))[0]
    str_last_command = list(db["control"].aggregate(
        [
            {'$sort' : {"_id" : -1}},
            {'$limit' : 1}
        ]
    ))[0]["command"]
        
    
    if ("end" in json_last_run) & (str_last_command == "stop"):
        return(True)
    else:
        return(False)
        

