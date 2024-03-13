import torch.nn as nn
import torch_geometric.nn as geom_nn
import model.Twitter.BiGCN_Twitter as bigcn
import model.Twitter.EBGCN as ebgcn
from lrp_pytorch.modules.linear import LRPLinear, Linear_Epsilon_Autograd_Fn, CAM_Linear_Autograd_Fn
from lrp_pytorch.modules.gcn_conv import LRPGCNConv, GCNConv_Autograd_Fn, CAM_GCNConv_Autograd_Fn, \
    EB_GCNConv_Autograd_Fn
from lrp_pytorch.modules.bigcn import LRPBiGCNRumourGCN, BiGCNRumourGCN_Autograd_Fn, LRPBiGCN, CAM_BiGCN, EB_BiGCN
from lrp_pytorch.modules.ebgcn import LRPEBGCNRumourGCN, EBGCNRumourGCN_Autograd_Fn, LRPEBGCN, CAM_EBGCN, EB_EBGCN
from copy import deepcopy


class Constants:

    key2class = {'nn.Linear': LRPLinear,
                 'geom_nn.GCNConv': LRPGCNConv,
                 'bigcn.TDrumorGCN': LRPBiGCNRumourGCN,
                 'bigcn.BUrumorGCN': LRPBiGCNRumourGCN,
                 'bigcn.BiGCN': LRPBiGCN,
                 'ebgcn.TDrumorGCN': LRPEBGCNRumourGCN,
                 'ebgcn.BUrumorGCN': LRPEBGCNRumourGCN,
                 'ebgcn.EBGCN': LRPEBGCN,
                 'nn.Linear(CAM)': LRPLinear,
                 'geom_nn.GCNConv(CAM)': LRPGCNConv,
                 'bigcn.BiGCN(CAM)': CAM_BiGCN,
                 'bigcn.BiGCN(EB)': EB_BiGCN,
                 'ebgcn.EBGCN(CAM)': CAM_EBGCN,
                 'ebgcn.EBGCN(EB)': EB_EBGCN,
                 }
    key2autograd_fn = {'nn.Linear': Linear_Epsilon_Autograd_Fn,
                       'geom_nn.GCNConv': GCNConv_Autograd_Fn,
                       'bigcn.TDrumorGCN': BiGCNRumourGCN_Autograd_Fn,
                       'bigcn.BUrumorGCN': BiGCNRumourGCN_Autograd_Fn,
                       'ebgcn.TDrumorGCN': EBGCNRumourGCN_Autograd_Fn,
                       'ebgcn.BUrumorGCN': EBGCNRumourGCN_Autograd_Fn,
                       'nn.Linear(CAM)': CAM_Linear_Autograd_Fn,
                       'geom_nn.GCNConv(CAM)': CAM_GCNConv_Autograd_Fn,}


def get_lrpwrappermodule(module, lrp_params, is_contrastive=False):
    if lrp_params.get('mode') == 'lrp':
        if isinstance(module, nn.Linear) or isinstance(module, geom_nn.Linear):
        # if isinstance(module, nn.Linear):
            key = 'nn.Linear'

            autograd_fn = Constants.key2autograd_fn[key]
            custom_class = Constants.key2class[key]
            return custom_class(module, autograd_fn, params=lrp_params)

        elif isinstance(module, geom_nn.GCNConv):
            key = 'geom_nn.GCNConv'

            autograd_fn = Constants.key2autograd_fn[key]
            custom_class = Constants.key2class[key]
            return custom_class(module, autograd_fn, params=lrp_params)

        elif isinstance(module, bigcn.TDrumorGCN):
            key = 'bigcn.TDrumorGCN'

            autograd_fn = Constants.key2autograd_fn[key]
            custom_class = Constants.key2class[key]
            return custom_class(module, autograd_fn, params=lrp_params)

        elif isinstance(module, bigcn.BUrumorGCN):
            key = 'bigcn.BUrumorGCN'

            autograd_fn = Constants.key2autograd_fn[key]
            custom_class = Constants.key2class[key]
            return custom_class(module, autograd_fn, params=lrp_params)

        elif isinstance(module, bigcn.BiGCN):
            key = 'bigcn.BiGCN'

            autograd_fn = None  # Constants.key2autograd_fn[key]
            custom_class = Constants.key2class[key]
            return custom_class(module, autograd_fn, params=lrp_params)

        elif isinstance(module, ebgcn.TDrumorGCN):
            key = 'ebgcn.TDrumorGCN'

            autograd_fn = Constants.key2autograd_fn[key]
            custom_class = Constants.key2class[key]
            return custom_class(module, autograd_fn, params=lrp_params)

        elif isinstance(module, ebgcn.BUrumorGCN):
            key = 'ebgcn.BUrumorGCN'

            autograd_fn = Constants.key2autograd_fn[key]
            custom_class = Constants.key2class[key]
            return custom_class(module, autograd_fn, params=lrp_params)

        elif isinstance(module, ebgcn.EBGCN):
            key = 'ebgcn.EBGCN'

            autograd_fn = None  # Constants.key2autograd_fn[key]
            custom_class = Constants.key2class[key]
            return custom_class(module, autograd_fn, params=lrp_params)

        else:
            print('Unknown module', module)
            return None
    elif lrp_params.get('mode') == 'cam':
        if isinstance(module, nn.Linear) or isinstance(module, geom_nn.Linear):
            # if isinstance(module, nn.Linear):
            key = 'nn.Linear(CAM)'

            autograd_fn = Constants.key2autograd_fn[key]
            custom_class = Constants.key2class[key]
            return custom_class(module, autograd_fn, params=lrp_params)

        elif isinstance(module, geom_nn.GCNConv):
            key = 'geom_nn.GCNConv(CAM)'

            autograd_fn = Constants.key2autograd_fn[key]
            custom_class = Constants.key2class[key]
            return custom_class(module, autograd_fn, params=lrp_params)

        elif isinstance(module, bigcn.TDrumorGCN):
            key = 'bigcn.TDrumorGCN(CAM)'

            autograd_fn = Constants.key2autograd_fn[key]
            custom_class = Constants.key2class[key]
            return custom_class(module, autograd_fn, params=lrp_params)

        elif isinstance(module, bigcn.BUrumorGCN):
            key = 'bigcn.BUrumorGCN(CAM)'

            autograd_fn = Constants.key2autograd_fn[key]
            custom_class = Constants.key2class[key]
            return custom_class(module, autograd_fn, params=lrp_params)

        elif isinstance(module, bigcn.BiGCN):
            key = 'bigcn.BiGCN(CAM)'

            autograd_fn = None  # Constants.key2autograd_fn[key]
            custom_class = Constants.key2class[key]
            return custom_class(module, autograd_fn, params=lrp_params)

        elif isinstance(module, ebgcn.TDrumorGCN):
            key = 'ebgcn.TDrumorGCN(CAM)'

            autograd_fn = Constants.key2autograd_fn[key]
            custom_class = Constants.key2class[key]
            return custom_class(module, autograd_fn, params=lrp_params)

        elif isinstance(module, ebgcn.BUrumorGCN):
            key = 'ebgcn.BUrumorGCN(CAM)'

            autograd_fn = Constants.key2autograd_fn[key]
            custom_class = Constants.key2class[key]
            return custom_class(module, autograd_fn, params=lrp_params)

        elif isinstance(module, ebgcn.EBGCN):
            key = 'ebgcn.EBGCN(CAM)'

            autograd_fn = None  # Constants.key2autograd_fn[key]
            custom_class = Constants.key2class[key]
            return custom_class(module, autograd_fn, params=lrp_params)

        else:
            print('Unknown module', module)
            return None
    elif lrp_params.get('mode') == 'eb':
        if isinstance(module, bigcn.BiGCN):
            key = 'bigcn.BiGCN(EB)'

            autograd_fn = None  # Constants.key2autograd_fn[key]
            custom_class = Constants.key2class[key]
            return custom_class(module, autograd_fn, params=lrp_params, is_contrastive=is_contrastive)

        elif isinstance(module, ebgcn.EBGCN):
            key = 'ebgcn.EBGCN(EB)'

            autograd_fn = None  # Constants.key2autograd_fn[key]
            custom_class = Constants.key2class[key]
            return custom_class(module, autograd_fn, params=lrp_params, is_contrastive=is_contrastive)

        else:
            print('Unknown module', module)
            return None
    else:
        print('Explainability method not specified')
        raise Exception
