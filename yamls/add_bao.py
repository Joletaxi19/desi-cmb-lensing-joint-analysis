import yaml
import sys
import os

with open(f'add_bao.yaml', 'r') as file: yamlData = yaml.safe_load(file)
yamlData['output'] = 'chains/'+str(sys.argv[1])
with open(f'add_bao_{str(sys.argv[1])}.yaml', 'w') as file: yaml.dump(yamlData,file)