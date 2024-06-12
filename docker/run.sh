#!/bin/bash
cd $(cd $(dirname $0); pwd)/..

WORK_DIR=/workspace
args=(
  --gpus all
  --shm-size=8g
  -it
  --rm
  -e LOCAL_UID=$(id -u $USER)
  -e LOCAL_GID=$(id -g $USER)
  -v ~/.gitconfig:/home/developer/.gitconfig
  -v ~/.ssh:/home/developer/.ssh:ro
  -v $(pwd):$WORK_DIR/hello-llm
)
docker run "${args[@]}" hello-llm
