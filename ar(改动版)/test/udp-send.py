import socket

def main():
    # 1.创建一个udp套接字
    udp_socket = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)

    # 2.准备接收方的地址
    # 192.168.65.149 表示目的地ip
    # 30000  表示目的地端口
    udp_socket.sendto("hello".encode("utf-8"), ("192.168.43.139", 30000))

    # 3.关闭套接字
    udp_socket.close()


if __name__ == "__main__":
    main()