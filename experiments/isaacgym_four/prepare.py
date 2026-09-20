"""Resolve the frozen four-run recipe without importing a simulator or starting W&B."""
import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import wandb
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parent
KEYS = ('jabs_finetune', 'eigendexplore_finetune', 'jabs_scratch', 'eigendexplore_scratch')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--weights-dir', type=Path, required=True)
    parser.add_argument('--basis', type=Path, required=True)
    parser.add_argument('--wandb-entity', required=True)
    parser.add_argument('--wandb-project', default='action-bench-simtoolreal')
    parser.add_argument('--wandb-group', default='isaacgym_four_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    parser.add_argument('--gpus', nargs=4, type=int, default=[0, 1, 2, 3])
    args = parser.parse_args()
    if len(set(args.gpus)) != 4 or min(args.gpus) < 0:
        parser.error('--gpus must name four different nonnegative physical GPU indices')
    for path in [args.basis, args.weights_dir / 'str_joint_abs_s13.pth',
                 args.weights_dir / 'str_joint_abs_eignoise_espp_hot_s202.pth']:
        if not path.is_file():
            parser.error('Required input is missing: ' + str(path))
    output = args.output_dir.resolve()
    if output.exists():
        parser.error('--output-dir must be new; existing run IDs/configs are never overwritten')
    os.environ.update(STR_RUN_ROOT=str(args.run_root.resolve()),
                      STR_WEIGHTS_DIR=str(args.weights_dir.resolve()),
                      STR_EIGEN_BASIS=str(args.basis.resolve()),
                      STR_WANDB_ENTITY=args.wandb_entity,
                      STR_WANDB_PROJECT=args.wandb_project,
                      STR_WANDB_GROUP=args.wandb_group)
    configs, manifest = [], []
    for key, gpu in zip(KEYS, args.gpus):
        os.environ['STR_WANDB_ID'] = wandb.util.generate_id()
        cfg = OmegaConf.to_container(OmegaConf.load(ROOT / 'configs' / (key + '.yaml')), resolve=True)
        cfg['campaign']['gpu'] = gpu
        run_dir = Path(cfg['campaign']['run_dir'])
        if run_dir.exists():
            parser.error('Run directory already exists: ' + str(run_dir))
        path = output / 'configs' / (key + '.yaml')
        configs.append((path, cfg))
        manifest.append(dict(cfg['campaign'], seed=cfg['seed'], config=str(path),
                             exploration_scale=cfg['train']['params']['config']['expl_reward_coef_scale'],
                             initial_learning_rate=cfg['train']['params']['config']['learning_rate'],
                             success_tolerance=cfg['task']['env']['successTolerance'],
                             wandb_url='https://wandb.ai/{}/{}/runs/{}'.format(
                                 args.wandb_entity, args.wandb_project, cfg['campaign']['wandb_id'])))
    (output / 'configs').mkdir(parents=True)
    for path, cfg in configs:
        OmegaConf.save(OmegaConf.create(cfg), path)
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(output / 'manifest.json')


if __name__ == '__main__':
    main()
