"""Portable input paths shared by the manifold analysis scripts."""

import argparse
from pathlib import Path


def default_inputs(include_train=True):
    """Return project-relative defaults without reading or loading any files."""
    root = Path(__file__).resolve().parent.parent
    data = root / 'model' / 'data'
    splits = {'te_r': ('real', 'test'), 'te_f': ('fake', 'test')}
    if include_train:
        splits = {'tr_r': ('real', 'train'), **splits}
    return {
        'paths': {key: str(data / 'metapath_data' / f'Deepseek_{label}_{split}_metapaths.txt')
                  for key, (label, split) in splits.items()},
        'bert_paths': {key: str(data / 'bert_feature' / f'{label}_{split}_bert.npy')
                       for key, (label, split) in splits.items()},
        'model_ckpt': str(root / 'model' / 'checkpoints' / 'SFTF_weibo25_best.pth'),
    }


def configure_inputs(config, argv=None):
    """Override and validate input paths before inference; leave other settings intact."""
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description='Run manifold analysis with existing features and weights.')
    parser.add_argument('--data-dir', type=Path, default=root / 'model' / 'data',
                        help='Directory containing metapath_data/ and bert_feature/ feature files (default: SFTF/model/data).')
    parser.add_argument('--checkpoint', type=Path, default=Path(config['model_ckpt']),
                        help='Compatible model state dictionary (default: SFTF/model/checkpoints/SFTF_weibo25_best.pth).')
    args = parser.parse_args(argv)
    data_dir = args.data_dir.expanduser().resolve()
    inputs = {
        'paths': {key: str(data_dir / 'metapath_data' / Path(value).name)
                  for key, value in config['paths'].items()},
        'bert_paths': {key: str(data_dir / 'bert_feature' / Path(value).name)
                       for key, value in config['bert_paths'].items()},
        'model_ckpt': str(args.checkpoint.expanduser().resolve()),
    }
    required = [*inputs['paths'].values(), *inputs['bert_paths'].values(), inputs['model_ckpt']]
    missing = [path for path in required if not Path(path).is_file()]
    if missing:
        parser.error('Missing input files:\n  ' + '\n  '.join(missing) +
                     '\nProvide --data-dir and --checkpoint for your existing files.')
    config.update(inputs)
