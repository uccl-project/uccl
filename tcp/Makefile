.PHONY: client server clean all local

CC=g++
CFLAGS=-I. -O3 -Wall -pedantic -g
DEPS = connection.h
PORT=12345

%.o: %.c $(DEPS)
		$(CC) -c -o $@ $< $(CFLAGS)

all: client server

client: client.o
		$(CC) -o $@ $^ $(CFLAGS)

server: server.o
		$(CC) -o $@ $^ $(CFLAGS)

clean:
	rm -f *.o client server

local: client server
	./server $(PORT) &
	./client localhost $(PORT)

