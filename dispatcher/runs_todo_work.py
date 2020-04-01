from pymongo import MongoClient
import os
import time
from datetime import datetime
import requests

daq_status_compare = {
    "daq":{"status": "offline", "active": False},
    "daqspatcher":{"active": True, "status": "online"},
    "straxinator":{"active": True, "status": "idle"},
    "pulser":{},
}





def bool_last_run_finished(db, verbose = False):
    try:
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
        json_system_states = list(db["system_control"].find())
        
        bool_system_states = True
        
        for json_system_state in json_system_states:
            list_compare = daq_status_compare[json_system_state["subsystem"]]
            
            for field_compare in list_compare:
                if not json_system_state[field_compare] == list_compare[field_compare]:
                    bool_system_states = False
                    
        
        int_upcoming_runs = db["runs_todo"].count_documents({})
        if verbose:
            print(
                '\n"end" in json_last_run:\n  ' + str(("end" in json_last_run))+
                '\nstr_last_command == "stop":\n  ' + str((str_last_command == "stop"))+
                '\nint_upcoming_runs > 0:\n  ' + str((int_upcoming_runs > 0))+
                '\nbool_system_states:\n  ' + str(bool_system_states)
            )
            
            
        if ("end" in json_last_run) & (str_last_command == "stop") & (int_upcoming_runs > 0) & bool_system_states:
            return(True)
        else:
            return(False)
            
    except:
        return(False)

def copy_next_run(db, verbose = False):
    json_next_run = list(db["runs_todo"].aggregate(
            [
                {'$sort' : {"_id" : 1}},
                {'$limit' : 1}
            ]
    ))[0]
    
    if json_next_run["mode"] == "led":
        json_next_run["goal"] = "led"
    

    json_next = {
        "comment" : "",
        "config_override": {},
        "duration": 300,
        "goal": "arm",
        "mode": "led",
        "user" : "runlist",
    }
    
    _id = json_next_run["_id"]
    if verbose:
       print(_id) 
    
    for key in json_next:
        if key in json_next_run:
            json_next[key] = json_next_run[key]
    
    if verbose:
       print(json_next) 
    
    
    # modify 
    try:
        if verbose:
            print("1")
        
        # start via post
        client = requests.session()
        client.get("http://localhost/control")

        json_next["csrfmiddlewaretoken"] = client.cookies.get_dict()["csrftoken"]

                
        command_request = client.post(
            "http://localhost/control/start",
            data = json_next,
            cookies = client.cookies
        )

        
        if verbose:
            print("2")
    
        db["runs_todo"].delete_one({"_id": _id})
    
        if verbose:
            print("3")
    
        return(True)
    except:
        return(False)
        
    
    
    
