"""Functions .xy file loading for Baldini Lab Momentum Microscope at UT Austin"""

"""It is convenient to be able to access individual functions for certain meta data
processing, along with being able to load multiple 
"""

import numpy as np
import xarray as xr
import re
from collections import OrderedDict

RENAME_COORDS = {
    "X": "x",
    "Y": "y",
    "Z": "z",

    #Conversion into angle name coordinates is necessary
    #for the PyARPES architecture. Will always notify when
    #the angle displayed is not actually the physically
    #correct coordinate. 
    "k_x": "k",
    "k_y": "ky",
    "kx": "kx",
    "ky": "ky",
    "x": "x",
    "y": "y",
    "z": "z",

    "Kinetic Energy": "eV",
    "energy": "eV"
}

def filter_meta_data(ls: list) -> dict:
    """Helps with text processing in load_xy function.
    Sorts list of strings into key:value pairs"""
    ls = list(map(lambda string: re.split(r'(\s\s)+', string), ls))
    ls_dict = {}
    for i in range(len(ls)):
        ls[i] = [string for string in ls[i] if not re.fullmatch(r'\s*', string)]
        ls[i] = [re.sub('^#', '', string) for string in ls[i]]
        
        if len(ls[i]) > 2:
            ls[i].pop(0)
        
        ls[i] = [re.sub(r'\s+', '', string) for string in ls[i]]
            
        try:
            ls_dict[ls[i][0]] = ls[i][1]
        except IndexError:
            None
    return ls_dict
    
def load_xy(file_path: str, ONLY_SETTINGS=False, ONLY_GROUPNAMES=False):
    """Loads .xy file from SPECS MM setup in Baldini Lab into Python dictionary format.
    Output is mostly used in 'load_xarray' function to extract data from 
    a specific trial run(EDC, FS, etc.), though could be helpful for debugging.

    Most of the complexity here comes from the fact that occasionally it's useful to have multiple
    full scanes (not just frames) in one .xy file. Thus this first parsing function is necessary.
    
    Args:
        file_path: The SPECS output file (MUST BE IN .xy FORMAT)
        ONLY_SETTINGS: If true will output only the experiment universal settings (useful for documentation)
        ONLY_GROUPNAMES: If true will outut only names of each group (used in other functions)"""

    try:
        with open(file_path) as file:
            lines = file.readlines()
    except UnicodeDecodeError:
        print("""There's some character that readlines can't deal with. Most likeley it's a 
        greek letter µ, or an approximation sign. Delete it or change the character
        inside the file to resolve this issue. Should only have maximum 2 instances in the file.""")
        return 1

    lines = [l.strip() for l in lines]
    settings_index = lines.index("#   Time Zone Format:         UTC")
    settings = filter_meta_data(lines[:settings_index+1])

    if ONLY_SETTINGS:
        return settings

    #GROUP DEPENDENT DATA AND METADATA
    meta_data = lines[settings_index+2:] #Everything after the settings
    raw_data = np.loadtxt(file_path)

    group_count = sum('Group' in string for string in meta_data)
    group_indexes = [index for index, string in enumerate(meta_data) if 'Group' in string]
    group_names = []
    for index in group_indexes:
        idx = meta_data[index].rfind('    ')
        group_names.append(meta_data[index][idx+1:].strip())
        
    if ONLY_GROUPNAMES == True:
        return group_names
        
    group_datas = []
    for i in range(len(group_indexes)):
        try:
            group_datas.append(meta_data[group_indexes[i]:group_indexes[i+1]])
        except IndexError:
            group_datas.append(meta_data[group_indexes[i]:])

    groups = dict(zip(group_names, group_datas))
    trial_settings = []
    
    for key in groups.keys():
        groups[key].pop(-1) #There was whitespace at end of each list
        group = groups[key]
        
        #OR for "OrdinateRange", and R for 'Region'. Slices out group dependent settings values
        group_OR_indexes = [index+3 for index, string in enumerate(group) if 'OrdinateRange:' in string] #Stripping messes with the indexes a bit, hence the + 
        group_R_indexes = [index for index, string in enumerate(group) if 'Region:' in string]

        #Just the data sets with their numbered labels
        #And filtering out group dependent meta data besides numbered labels
        trials = []
        run_settings = dict()
        if len(group_OR_indexes) == 1:
            trials.append(group[group_OR_indexes[0]:])
            run_settings['Trial 1'] = filter_meta_data(group[:group_OR_indexes[0]])
        else:
            for i in range(len(group_OR_indexes)):
                run_settings[f'Trial {i+1}'] = filter_meta_data(group[group_R_indexes[i]:group_OR_indexes[i]])
                try:                     
                    trials.append(group[group_OR_indexes[i]:group_R_indexes[i+1]-1])
                except IndexError:
                    trials.append(group[group_OR_indexes[i]:])    

        trial_datas = dict()
        count = 0

        for i, trial in enumerate(trials):
            cycle_curve_dict = dict()
            cycle_curve_data = []
            cycle_curve_params = []
            parameter = ""
            coordinates = ''
            multiple_channels = True
            
            #Initializing for first parse through
            if 'yes' in settings["SeparateChannelData:"]:
                cycle_names = ['Cycle: 0, Curve: 0, Channel: 0'] 
            else: 
                cycle_names = ['Cycle: 0, Curve: 0']
                
            for line in trial:
                try:
                    if "Parameter:" in line:
                        parameter = re.sub(': ', ':   ', line)
                    if "Curve:" in line:
                        cycle_names.append(line[2:])
                        cycle_curve_params.append(parameter) 
                        cycle_curve_dict[cycle_names[0]] = [
                            filter_meta_data(cycle_curve_params),
                            np.array(cycle_curve_data)
                        ]
                        
                        #Making sure coordinates have a space in between them
                        if len(cycle_curve_params) > 4:
                            #Note, I leave it as a string, as there may be whitespace in the coordinate names
                            coordinates = cycle_curve_params[-3].split('   ')[-1].strip()
                        elif len(cycle_curve_params) > 3:
                            #Covering case where there is no parameter
                            
                            coordinates = cycle_curve_params[-2].split('   ')[-1].strip()
                            
                        cycle_names.pop(0)
                        cycle_curve_data, cycle_curve_params = [], []
                    if line[0] == '#':
                        cycle_curve_params.append(line[2:])
                    elif line[0] != '#':
                        numbers = line.split('  ')
                        cycle_curve_data.append([float(numbers[0]), float(numbers[1])])
                except IndexError:
                    None

            cycle_curve_params.append(parameter) if parameter != "" else None
            cycle_curve_dict[cycle_names[-1]] = [filter_meta_data(cycle_curve_params), np.array(cycle_curve_data)]
            
            #Making sure coordinates have a space in between them
            for k in cycle_curve_dict.keys():
                cycle_curve_dict[k][0]['ColumnLabels:'] = coordinates
                
            trial_datas[f'Trial {i+1}'] = cycle_curve_dict

        for k in run_settings.keys():
            run_settings[k].update(trial_datas[k])
        
        groups[key] = run_settings

    return {
        'settings': settings,
        'groups': groups,
    }

def load_to_xarray(data: dict, group_name: str = None, trial_name: str = None):
    """Takes output of load_xy function and outputs xarray of desired trial.
    Args:
        data: The whole output of load_xy(file)
        group_name: Name of desired group
        trial_name: Name of desired trial within group"""

    #Error handling if group_name or trial_name is invalid
    if group_name != None:
        try:
            group = data['groups'][group_name]
        except KeyError:
            groups  = []
            for group in data['groups'].keys():
                groups.append(group)
            print(f"""Group name not in data provided. Please choose from one of the following groups:
            {groups}""")
            return 1
    elif trial_name != None:
        try:
            trial = group[trial_name]
        except KeyError:
            trials  = []
            for trial in group.keys():
                trials.append(trial)
            print(f"""Trial name not in data provided. Please choose from one of the following trials:
            {trials}""")
            return 2
    else:
        name = str(list(data['groups'].keys())[0])
        group = data['groups'][name]
        trial = group['Trial 1']

    #Updating universal settings with trial specific settings
    settings = data['settings']
    for key, value in trial.items():
        if type(trial[key]) == list:
            continue
        else:
            settings.update({key:value})

    #Loading data into numpy array
    cuts = []
    array = []
    arrays = []
    cuts = []
    cycle = 0
    curve = 0
    one_cut = True
    one_curve = True
    for key in trial.keys():
        if 'Cycle: 1' in key:
            one_cut = False
            break
        elif 'Curve: 1, Channel: 0' in key:
            one_curve = False
    
    for key, value in trial.items():
        if type(value) != list or len(trial[key][1]) == 0:
            continue
        
        #Only 1 cut
        if one_cut == True:
            if one_curve == True:
                array.append(trial[key][1])
            else:
                if int(re.sub(',', '', key[17:20])) == curve:
                    array.append(trial[key][1])
                elif int(re.sub(',', '', key[17:20])) == curve+1:
                    curve += 1
                    cuts.append(array)
                    array = [trial[key][1]]
            continue

        #More than one cut
        #NOTE: Haven't yet implemented both more than one cut AND separate channel data
        else:
            if int(re.sub(',', '', key[7:9])) == cycle:
                array.append(trial[key][1])
            elif int(re.sub(',', '', key[7:9])) == cycle+1:
                cycle += 1
                cuts.append(array)
                array = [trial[key][1]]

    cuts.append(array) #Appending final array as the loop doesn't get to it
    scan_var_coords = np.array(cuts)[0, 0, :, 0] #Easiest to get it here, even if it's a bit ouf of place
    cuts = np.array(cuts)[:, :, :, 1] #0th dimension is just the scan variable

    #BEGIN LOADING INTO XARRAY (with calibration)
    scan_var = trial[list(trial.keys())[-1]][0]['ColumnLabels:'].split(' ')[0].strip()
    try:
        scan_var = RENAME_COORDS[scan_var]
    except KeyError:
        print("Note: Scan coordinate not renamed to PyARPES convention. Certain functionalities may not work")

    """
    Not correct, leaving in for debugging
    scan_var_coords = settings['OrdinateRange:'].strip()[1:len(settings['OrdinateRange:'])-1].split(',')
    scan_var_coords = [float(val) for val in scan_var_coords]
    """
    scan_var_coords = scan_var_coords #Just to be consistent
    
    #Interpreting NonEnergyOrdinate from imaging mode
    if 'MM_Momentum' in settings['AnalyzerLens:']:
        real_non_energy_ordinate = 'ky'
    elif 'MM_PEEM' in settings['AnalyzerLens:']:
        real_non_energy_ordinate = 'y'
    elif 'ARPES' in settings['AnalyzerLens:']:
        real_non_energy_ordinate = 'ky'

    try:
        non_energy_ordinate = RENAME_COORDS[real_non_energy_ordinate]
        non_energy_ordinate_coords = []

        for value in trial.values():
            if type(value[0]) != dict:
                continue
            non_energy_ordinate_coords.append(float(value[0]['NonEnergyOrdinate:']))
        non_energy_ordinate_coords = list(OrderedDict.fromkeys(non_energy_ordinate_coords))
    
    except KeyError:
        non_energy_ordinate = "No_NEO"
        non_energy_ordinate_coords = 0

    #Channel Coordinate
    if 'yes' in settings['SeparateChannelData:']:
        if 'MM_Momentum' in settings['AnalyzerLens:']:
            real_channel = 'kx'
        elif 'MM_PEEM' in settings['AnalyzerLens:']:
            real_channel = 'x'
        elif 'ARPES' in settings['AnalyzerLens:']:
            real_channel = 'kx'
        channel = RENAME_COORDS[real_channel]
        channel_coords = non_energy_ordinate_coords #Until we can find out how to access them
            
    else:
        channel = "No_Channel"
        channel_coords = 0
            
    
    #Logical Variable Calibration
    try:
        logical_var = re.sub('"', '', trial[str(list(trial.keys())[-1])][0]['Parameter:'].split('=')[0])
        logical_var_coords = []
        
        for i, k in enumerate(trial.keys()):
            if len(trial[k][0]) > 2 and float(trial[k][0]['Parameter:'].split('=')[1]) not in logical_var_coords:
                logical_var_coords.append(float(trial[k][0]['Parameter:'].split('=')[1]))
    except KeyError:
        logical_var = "No_LV"
        logical_var_coords = 0
    
    #Getting rid of whitespace in coordinate names
    logical_var = re.sub(r'\s+','_',logical_var.strip())
    logical_var = re.sub(r'[\[\]]','_',logical_var)

    #Other values which correspond to coordinates in the PyARPES model
    
    array_dims = []
    if one_cut == False:
        array_dims.append(logical_var)
    if 'yes' in settings['SeparateChannelData:']:
        array_dims.append(channel)
    array_dims.append(non_energy_ordinate)
    array_dims.append(scan_var)

    spectrum_coords = {
            scan_var: scan_var_coords,
            non_energy_ordinate: non_energy_ordinate_coords,
            channel: channel_coords,
            logical_var: logical_var_coords,

            #Non scannable coordinates
            "chi": 0,
            "theta": 0,
            "psi": 0,
            "alpha": 0
        }
    
    spectrum = xr.DataArray(
        data = cuts,
        dims = array_dims,
        coords = spectrum_coords,
        attrs = settings
    )
    
    return xr.Dataset(
        {'spectrum': spectrum}, attrs = settings
    )

def load_xy_data(file_name: str, group_name: str = None, trial_name: str = None):
    """Basically just a wrapper around 'load_xy' and 'load_to_xarray' functions.
    Provides convenience for loading"""
    
    data = load_xy(file_name)

    data = load_to_xarray(data, group_name=group_name, trial_name=trial_name)

    return data
