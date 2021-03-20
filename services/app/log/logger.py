import os
import sys
import log4p
import json

# os.path.dirname(os.path.realpath(__file__))
# file_path = os.path.dirname(os.path.realpath(__file__)) + '/log4p.json'
with open('log/log4p.json') as f:
  data = json.load(f)
dir = data["handlers"]["file_handler"]["filename"]
log_dir = dir[:-9]
# filename = dir[-8:]

if not os.path.exists(log_dir):
    os.makedirs(log_dir)
    f = open(dir, mode='wt', encoding='utf-8')
    f.close()

log = log4p.GetLogger(__name__, config=os.getcwd()+'/log/log4p.json')
print('log',log)
logger = log.logger