import json
import os

def default_args():
    default_path = os.path.dirname(os.path.realpath(__file__))
    return load_argsfile(os.path.join(default_path, 'default_args.json'))

def combine(da, db):
    """
    description:
    combine two dicts with priority to db.
    """
    import copy
    da = copy.deepcopy(da) 
    return da.update(db)

def load_argsfile(argfile_path):
    with open(argfile_path) as fp:
        _args = json.load(fp)
    return _args

def load_args(arg_path):
    if arg_path != '':
        args = load_argsfile(arg_path)
    else:
        args = default_args()
    #combined = combine(_default_args, _usr_args)
    return args

def logdictargs(fullpath, args):
    import json
    with open(fullpath, 'w') as fp:
        fp.write(json.dumps(args, indent=4))
