/*
 *  A server receiving and sending back a message multiple times.
 *  Usage: ./server.out -p <port> -n <message_size (bytes)>
 */
#include "util_tcp.h"
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <fcntl.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <strings.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

void error(char* msg) {
  perror(msg);
  exit(1);
}

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
  listen(sockfd, 5);
  socklen_t clilen = sizeof(cli_addr);

  // Accept connection and set nonblocking and nodelay
  newsockfd = accept(sockfd, (struct sockaddr*)&cli_addr, &clilen);
  if (newsockfd < 0) {
    error("ERROR on accept");
  }
  fcntl(newsockfd, F_SETFL, O_NONBLOCK);
  setsockopt(newsockfd, IPPROTO_TCP, TCP_NODELAY, (void*)&flag, sizeof(int));

  // Receive-send loop
  printf("Connection accepted, ready to receive!\n");
  for (size_t i = 0;; i++) {
    receive_message(config.n_bytes, newsockfd, buffer, &quit);
#ifdef PING_PONG_MSG
    send_message(config.n_bytes, newsockfd, buffer, &quit);
#else
    send_message(sizeof(uint64_t), newsockfd, buffer, &quit);
#endif
  }
  printf("Done!\n");

  // Clean state
  close(sockfd);
  close(newsockfd);

  return 0;
}
