#!/bin/bash
cd $(cd $(dirname $0); pwd)/..

docker build -t hello-llm -f docker/Dockerfile .
