import yaml
import matplotlib.pyplot as plt
import pprint

import lib
from lib import nomenclature
from lib.arg_utils import define_args
from lib.utils import load_model, load_model_by_name
from lib.loggers import NoLogger

_t = {
    'crossformer': 'CrossFormer',
    'vit': 'SimpleViT',
    'twins-svt': 'Twins-SVT',
    't2t': 'Token2Token',
    'cait': 'CaiT',
}

args = define_args(
    extra_args = [
        ('--checkpoint', {'default': '', 'type': str, 'required': True})
    ])

architecture = nomenclature.MODELS[args.model](args)

state_dict = load_model_by_name(args.checkpoint, kind = 'best')
state_dict = {
    key.replace('module.', ''): value
    for key, value in state_dict.items()
}

architecture.load_state_dict(state_dict)
architecture.eval()
architecture.train(False)
architecture.to(lib.device)

evaluator = nomenclature.EVALUATORS['casia-recognition'](args, architecture, logger = NoLogger())
fig, ax = evaluator.visualize()
ax.set_title(_t[args.model])
print('saving model to ', args.checkpoint)
plt.tight_layout()
plt.savefig('figs/' + args.checkpoint)
# plt.show()
