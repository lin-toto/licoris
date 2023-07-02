#ifndef EMBEDLIC_NETWORK_H
#define EMBEDLIC_NETWORK_H

#include <string>
#include <stdexcept>
#include <cstring>
#include <unistd.h>

#ifdef _WIN32
#include <winsock2.h>
#include <windows.h>
#else
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#endif

class TCPSender {
public:
    TCPSender(std::string ip, unsigned int port);
    ~TCPSender();
    TCPSender(const TCPSender&) = delete;
    TCPSender& operator=(TCPSender) = delete;
    void send(const std::string& content) const;
private:
    uint64_t socketFd;
};

class TCPReceiver {
public:
    TCPReceiver(std::string bind, unsigned int port);
    ~TCPReceiver();
    TCPReceiver(const TCPReceiver&) = delete;
    TCPReceiver& operator=(TCPReceiver) = delete;
    void waitClient();
    std::string receive() const;
private:
    uint64_t connectFd = 0;
    uint64_t socketFd;
};

#endif //EMBEDLIC_NETWORK_H
