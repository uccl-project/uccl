/*
 *  A server receiving and sending back a message multiple times.
 *  Usage: ./server.out -p <port> -n <message_size (bytes)>
 */
#include "util_tcp.h"
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <assert.h>
#include <fcntl.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <strings.h>
#include <sys/epoll.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

void error(char* msg) {
  perror(msg);
  exit(1);
}

int const MY_RECV_BATCH_SIZE = 32;

int main(int argc, char* argv[]) {
  int sockfd, newsockfd;
  struct Config config = get_config(argc, argv);
  uint8_t* buffer = (uint8_t*)malloc(config.n_bytes);
  struct sockaddr_in serv_addr, cli_addr;

  // Create listening socket
  sockfd = socket(AF_INET, SOCK_STREAM, 0);
  if (sockfd < 0) {
    error("ERROR opening socket");
  }
  int flag = 1;
  if (setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &flag, sizeof(int)) < 0)
    error("setsockopt(SO_REUSEADDR) failed");

  bzero((char*)&serv_addr, sizeof(serv_addr));
  serv_addr.sin_family = AF_INET;
  serv_addr.sin_addr.s_addr = INADDR_ANY;
  serv_addr.sin_port = htons(config.port);
  if (bind(sockfd, (struct sockaddr*)&serv_addr, sizeof(serv_addr)) < 0) {
    error("ERROR on binding");
  }

  printf("Server ready, listening on port %d\n", config.port);
  fflush(stdout);
  listen(sockfd, NUM_SOCKETS * 2);
  socklen_t clilen = sizeof(cli_addr);

  for (int i = 0; i < NUM_SOCKETS; i++) {
    // Accept connection and set nonblocking and nodelay
    newsockfd = accept(sockfd, (struct sockaddr*)&cli_addr, &clilen);
    if (newsockfd < 0) {
      error("ERROR on accept");
    }
    fcntl(newsockfd, F_SETFL, O_NONBLOCK);
    setsockopt(newsockfd, IPPROTO_TCP, TCP_NODELAY, (void*)&flag, sizeof(int));
    config.sockfds[i] = newsockfd;
  }

  int epfd = epoll_create(1);

  // Monitor all sockets for EPOLLIN
  for (int i = 0; i < NUM_SOCKETS; i++) {
    struct epoll_event ev;
    ev.events = EPOLLIN | EPOLLEXCLUSIVE;
    ev.data.fd = config.sockfds[i];

    if (epoll_ctl(epfd, EPOLL_CTL_ADD, config.sockfds[i], &ev) == -1) {
      perror("epoll_ctl()\n");
      exit(EXIT_FAILURE);
    }
  }

  // Receive-send loop
  printf("Connection accepted, ready to receive!\n");
  struct epoll_event events[MAX_EVENTS_ONCE];
  while (!quit) {
    int nfds = epoll_wait(epfd, events, MAX_EVENTS_ONCE, -1);
    for (int i = 0; i < nfds; i++) {
      assert(events[i].events & EPOLLIN);
      int recv_cnt = 0;
      for (; recv_cnt < MY_RECV_BATCH_SIZE; recv_cnt++) {
        int ret = receive_message_early_return(
            config.n_bytes, events[i].data.fd, buffer, &quit);
        if (ret == 0) {  // would block
          break;
        }
        send_message(config.n_bytes, events[i].data.fd, buffer, &quit);
      }
    }
  }
  printf("Done!\n");

  // Clean state
  close(sockfd);
  for (int i = 0; i < NUM_SOCKETS; i++) {
    close(config.sockfds[i]);
  }

  return 0;
}
