CTRL_NIC="eth0"

mpirun --bind-to none -np 2 \
    -hostfile hostname.txt \
    --mca btl_tcp_if_include ${CTRL_NIC} \
    --mca plm_rsh_args "-o StrictHostKeyChecking=no" \
    --mca orte_base_help_aggregate 0 \
    -x GLOG_logtostderr=1 \
    -x GLOG_v=0 \
    ./incast