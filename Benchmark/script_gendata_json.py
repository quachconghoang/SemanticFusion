import os
import json
from SlamUtils.Loader.TartanAir import rootDIR, getDataSequences, getDataLists, tartan_camExtr

def genFullSequences(path_scr = rootDIR):
    TartanAir_scenarios = os.listdir(path_scr)
    dict_full = {}
    for scene in TartanAir_scenarios:
        if os.path.isdir(os.path.join(path_scr,scene)) == False:
            TartanAir_scenarios.remove(scene)

    dict_full.update({'keys': TartanAir_scenarios})
    dict_full.update({'levels': ['Easy', 'Hard']})
    for scene in TartanAir_scenarios:
        _dict_scene = {}
        _path_sce = os.path.join(path_scr,scene)
        levels = os.listdir(_path_sce); levels.sort()
        for lvl in levels:
            _trajs = os.listdir(os.path.join(_path_sce,lvl))
            _trajs.sort()
            _traj_paths = []
            for _traj in _trajs:
                # _path = os.path.join(_path_sce,lvl,_traj,'')
                _traj_paths.append(_traj)
            print(scene, '-', lvl)
            _dict_scene.update({lvl:_traj_paths})
        dict_full.update({scene:_dict_scene})
    # print(dict_full)
    return dict_full

myDict = genFullSequences()
with open(rootDIR + 'tartanair_data.json', 'w') as fp:
    json.dump(myDict, fp)

# with open(rootDIR + 'tartanair_data.json', 'r') as fp:
#     checkDict = json.load(fp)