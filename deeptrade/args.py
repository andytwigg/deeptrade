import os
import argparse

def common_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='load config from json file', default=None)
    parser.add_argument('--load_path', help='load from a specific checkpoint', default=None)
    parser.add_argument('--deterministic_step', action='store_true')
    parser.add_argument('--curses', action='store_true')
    parser.add_argument('--slow', action='store_true')
    parser.add_argument('--random', action='store_true')
    parser.add_argument('-v', '--verbose', help='verbose', action='store_true')
    parser.add_argument('--random_actions', action='store_true') # for eg curses
    return parser

def extract_args(args, char='.'):
    # {foo:1, abc.def: 1, abc.ghi: 2} => {foo:1, abc: {def: 1, ghi: 2}}
    d = {}
    for k,v in args.items():
        x=k.split(char,1)
        if len(x)==1:
            d[x[0]]=v
        else:
            if x[0] not in d:
                d[x[0]]={}
            d[x[0]][x[1]]=v
    return d

def to_pathspec(config):
    env_args, model_args = config['env'], config['model']
    print(env_args)
    s = '/'.join([
        config['env_id'],
        env_args['product_id'],
        str(env_args["step_type"]),
        str(env_args["step_val"]),
        config['policy'],
        'state_'+env_args["state_fn"],
        'rew_'+env_args["reward_fn"]+str(env_args["rew_eta"]),
        'seed_'+str(config['seed']),
    ])
    return s

def load_config(fname):
    import simplejson as json
    if fname is None:
        raise ValueError('config path cannot be None')
    print('loading config from {}'.format(fname))
    with open(fname,'rt') as fh:
        argsd = json.load(fh)
    args = extract_args(argsd)
    return args

def save_config(args, path):
    import simplejson as json
    fname = os.path.join(path, 'config.json')
    print('saving config to {}'.format(fname))
    with open(fname,'wt') as fh:
        json.dump(args, fh, indent=2)

def get_git_info():
    import subprocess, re, sys
    git_info = {'diff' : '', 'branch': '', 'url': ''}
    try:
        FNULL = open(os.devnull, 'w')
        git_info['diff'] = subprocess.Popen(['git', 'diff'], stdout=subprocess.PIPE, stderr=FNULL).communicate()[0].decode()
        size_mb = sys.getsizeof(git_info['diff']) / 1000000.
        if size_mb > 0.2:
            git_info['diff'] = "git diff too large to show here"
            print("Warning: git diff too large to track.")
        git_info['commit'] = subprocess.Popen(['git', 'log', 'HEAD', '-1'], stdout=subprocess.PIPE, stderr=FNULL).communicate()[0].decode()#.replace('\n', '')
        git_remote = subprocess.Popen(['git', 'remote', '-v'], stdout=subprocess.PIPE, stderr=FNULL).communicate()[0].decode().split()[1]
        git_info['url'] = git_remote #re.findall('\S*\.git', git_remote)[0]
    except Exception as e:
        pass
    return git_info

def dump_git_info(path):
    if path is None:
        return
    with open(os.path.join(path, 'git-info'), 'wt') as fh:
        for k,v in get_git_info().items():
            print('{} = {}'.format(k, v), file=fh)
