'''
Ḿisc utility functions
'''
import os
import json
import torch
import torch.backends.cudnn as cudnn

'''
 o8o               o8o      .
 `"'               `"'    .o8
oooo  ooo. .oo.   oooo  .o888oo
`888  `888P"Y88b  `888    888
 888   888   888   888    888
 888   888   888   888    888 .
o888o o888o o888o o888o   "888"
'''

def setup_env(args):
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        cudnn.benchmark = True


# create necessary folders and config files
def init_output_env(args):
    check_dir(os.path.join(args.data_dir,'runs'))
    check_dir(args.log_dir)
    check_dir(os.path.join(args.log_dir,'pics'))
    check_dir(os.path.join(args.log_dir,'tensorboard'))
    check_dir(os.path.join(args.log_dir,'pred'))
    # check_dir(os.path.join(args.log_dir, 'watch'))
    check_dir(args.res_dir)
    with open(os.path.join(args.log_dir, 'config.json'), 'w') as f:
        json.dump(args.__dict__, f)
    

'''
                   o8o
                   `"'
ooo. .oo.  .oo.   oooo   .oooo.o  .ooooo.
`888P"Y88bP"Y88b  `888  d88(  "8 d88' `"Y8
 888   888   888   888  `"Y88b.  888
 888   888   888   888  o.  )88b 888   .o8
o888o o888o o888o o888o 8""888P' `Y8bod8P'
'''

# check if folder exists, otherwise create it
def check_dir(dir_path):
    dir_path = dir_path.replace('//','/')
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)   


def save_res_list(res_list, fn):
    with open(fn, 'w') as f:
        json.dump(res_list, f)


def count_params(model):
   return sum([p.data.nelement() for p in model.parameters()])
