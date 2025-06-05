#!/bin/bash

for host in 192.168.102.191 192.168.102.192 192.168.102.193 192.168.102.194 192.168.102.195
do
  ssh root@$host "mkdir -p $UCCL_HOME"
  scp -r $UCCL_HOME root@$host:/root/
done