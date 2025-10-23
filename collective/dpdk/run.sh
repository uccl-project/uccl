ps aux | grep transport_test | grep -v grep | awk '{print $2}' | xargs sudo kill -9
make && scp transport_test 10.10.1.2:~/
