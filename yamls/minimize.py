import yaml
import sys
import os

yamlName    = str( sys.argv[1])

# set maximize = True 
with open(f'{yamlName}.yaml', 'r') as file: yamlData = yaml.safe_load(file)
keys = [key for key in yamlData['likelihood'].keys() if key.startswith('cobaya_friendly')]
for key in keys: yamlData['likelihood'][key]['maximize'] = True
# update name and output
yamlData['sampler'] = {'mcmc': None}
if yamlName.startswith('chains/'): 
    newYamlName        = f'{yamlName[7:]}_minimize'
    yamlData['output'] = 'chains/'+yamlData['output']+'_minimize' 
else: 
    newYamlName        = f'{yamlName}_minimize'
    yamlData['output'] = f'chains/{newYamlName}'
# and save
with open(f'{newYamlName}.yaml', 'w') as file: yaml.dump(yamlData,file)