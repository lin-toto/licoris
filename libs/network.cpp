#include "network.h"

TCPSender::TCPSender(std::string ip, unsigned int port) {
    socketFd = socket(AF_INET, SOCK_STREAM, 0);
    if (socketFd < 0) {
        throw std::runtime_error("Unable to create socket");
    }

    struct sockaddr_in address{};
    memset(&address, 0, sizeof(address));
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = inet_addr(ip.c_str());
    address.sin_port = htons(port);

    if (connect(socketFd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0) {
        throw std::runtime_error("Unable to connect");
    }
}

TCPSender::~TCPSender() {
    close(socketFd);
}

void TCPSender::send(const std::string& content) const {
    uint32_t length = htonl(content.length());

    if (::send(socketFd, reinterpret_cast<char *>(&length), sizeof(uint32_t), 0) < 0) {
        throw std::runtime_error("Error sending length data");
    }

    if (::send(socketFd, content.c_str(), content.length(), 0) < 0) {
        throw std::runtime_error("Error sending packet");
    }
}

TCPReceiver::TCPReceiver(std::string bind, unsigned int port) {
    socketFd = socket(AF_INET, SOCK_STREAM, 0);
    if (socketFd < 0) {
        throw std::runtime_error("Unable to create socket");
    }

    struct sockaddr_in address{};
    memset(&address, 0, sizeof(address));
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = inet_addr(bind.c_str());
    address.sin_port = htons(port);

    if (::bind(socketFd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0) {
        throw std::runtime_error("Bind failed");
    }

    if (listen(socketFd, 1) != 0) {
        throw std::runtime_error("Listen failed");
    }
}

TCPReceiver::~TCPReceiver() {
    if (connectFd) close(connectFd);
    close(socketFd);
}

void TCPReceiver::waitClient() {
    if (connectFd) throw std::runtime_error("Already connected");

    struct sockaddr_in client{};
    int len = sizeof(client);
    connectFd = accept(socketFd, reinterpret_cast<sockaddr*>(&client), &len);
}

std::string TCPReceiver::receive() const {
    if (!connectFd) throw std::runtime_error("Not connected");

    uint32_t length;
    if (recv(connectFd, reinterpret_cast<char *>(&length), sizeof(length), MSG_WAITALL) < 0) {
        throw std::runtime_error("Unable to read frame length");
    }
    length = ntohl(length);

    char *buffer = new char[length];

    int bytesRead = 0;
    while (bytesRead < length) {
        int count = recv(connectFd, buffer + bytesRead, length - bytesRead, MSG_WAITALL);
        if (count < 0) {
            throw std::runtime_error("Unable to read frame data");
        }

        bytesRead += count;
    }

    std::string result(buffer, length);
    delete[] buffer;

    return result;
}