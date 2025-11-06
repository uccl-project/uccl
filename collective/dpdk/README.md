sudo ./transport_test --logtostderr=1 --clientip=128.110.218.238 --test=bimq
sudo ./transport_test --logtostderr=1 --client --serverip=128.110.218.225 --test=bimq

sudo ./transport_test --logtostderr=1 --localip=172.168.0.1 --localmac=6c:92:bf:f3:2e:1a --test=mq
sudo ./transport_test --logtostderr=1 --client --localip=172.168.0.2 --localmac=6c:92:bf:f3:83:1e --serverip=11.11.11.39 --clientip=11.11.11.40 --test=mq

ps aux | grep transport_test | grep -v grep | awk '{print $2}' | xargs sudo kill -9

ps aux | grep test | grep -v grep | awk '{print $2}' | xargs sudo kill -9
rm transport_test && make transport_test && scp transport_test 11.11.11.40:~/

// Remember to set the NUM_QUEUE back to 2
