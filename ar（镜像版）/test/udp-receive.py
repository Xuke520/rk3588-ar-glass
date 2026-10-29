import socket

def main():
    udp_socket = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
    local_addr = ("", 30000)
    udp_socket.bind(local_addr)

    while True:
        recv_data = udp_socket.recvfrom(1024)  
        print(recv_data[0].decode('gbk'))
        print(recv_data[1])
    
    udp_socket.close()


if __name__ == "__main__":
    main()