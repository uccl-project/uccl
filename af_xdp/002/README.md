# 002

Let's send even more packets!

By using *two cores* we should be able to double the number of packets sent, right?

From this point forward, we want real CPU cores, not hyperthread cores on the same CPU.

Disable hyperthreading on both your client and server Linux boxes:

```console
echo off | sudo tee /sys/devices/system/cpu/smt/control
```

Now build and run the client and server, after modifying the source to match your interface name, IP address and ethernet addresses:

```
make && sudo ./server
```

```
make && sudo ./client
```

The results are disappointing:

```
sent delta 5960640
sent delta 6087904
sent delta 6095892
sent delta 6096036
sent delta 6108820
sent delta 6091460
sent delta 6092058
sent delta 6097242
sent delta 6093232
sent delta 6095776
```

It's _slightly_ slower than before, what's going on?
