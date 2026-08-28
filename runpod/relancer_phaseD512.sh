#!/bin/bash
# Relance la phase D-512 degelee apres une eviction du pod.
#
# Le conteneur perd /root, /tmp et les paquets pip a chaque redemarrage ; seul
# /workspace persiste. Ce script est donc deploye sur /workspace et se suffit a
# lui-meme : il reinstalle ce qui manque avant de lancer.
#
# La reprise se fait depuis run_dir/checkpoint.pt, que le trainer trouve avant
# resume_path. C'est pour cela que phaseD512_degele.yaml porte desormais
# resume_schedule: true et resume_optimizer: true -- la source est un
# checkpoint de la meme phase, donc compatible.
cd /workspace/code-memoire || exit 1
python3 /workspace/preflight.py || exit 1
export PYTHONPATH=deps:.
export WANDB_MODE=offline
C=configs/training/ciphermark/phaseD512_degele.yaml
nohup python3 distill.py config=$C \
  >> /workspace/logs/phaseD512_degele.log 2>&1 &
echo "entrainement lance, pid $!"
