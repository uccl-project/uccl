sudo ./transport_test --logtostderr=1 --clientip=128.110.218.238 --test=bimq
sudo ./transport_test --logtostderr=1 --client --serverip=128.110.218.225 --test=bimq

sudo ./transport_test --logtostderr=1 --test=mq
sudo ./transport_test --logtostderr=1 --client --serverip=128.110.218.225 --clientip=128.110.218.238 --test=mq

ps aux | grep transport_test | grep -v grep | awk '{print $2}' | xargs sudo kill -9
rm transport_test && make transport_test && scp transport_test 10.10.1.2:~/

// Remember to set the NUM_QUEUE back to 2
