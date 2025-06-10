#!/bin/bash

SOURCE_DIR="/home/ubuntu/uccl/rdma/incast"

TARGET_MACHINES=("87.120.213.6" "87.120.213.5")

TARGET_DIR="${SOURCE_DIR}"

if [ ! -d "$SOURCE_DIR" ]; then
  exit 1
fi

for MACHINE in "${TARGET_MACHINES[@]}"; do
  (
    echo "Installing on machine: $MACHINE"

    rsync -avz -e 'ssh -o StrictHostKeyChecking=no' --delete "$SOURCE_DIR/" "aleria@$MACHINE:$TARGET_DIR/" > /dev/null 2>&1

    if [ $? -eq 0 ]; then
      echo "Copy done on machine: $MACHINE"

      ssh "aleria@$MACHINE" "cd $TARGET_DIR && make clean && make" > /dev/null 2>&1
      if [ $? -eq 0 ]; then
        echo "Compile successfully on machine: $MACHINE"
      else
        echo "Compile error on machine: $MACHINE"
      fi
    else
      echo "Can't access machine: $MACHINE"
    fi
  ) &
done

wait

echo "Done."