# 20191118 그리퍼 소캣통신

import socket
import time

HOST1 = "192.168.10.72"
HOST2 = "192.168.10.77"
socket_ip1 = HOST1  # 오른쪽 팔
socket_ip2 = HOST2  # 왼쪽 팔
PORT = 63352


class GripperSocket:
    def __init__(self, socket_ip=HOST2, socket_port=PORT):
        print("Starting communicate")
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.connect((socket_ip, socket_port))
        str1 = b"GET GTO\n"
        self.s.send(str1)
        print("Q : {}".format(str1.decode()), end="")
        a = self.s.recv(1024)
        print("A : {}".format(a.decode()))

    def get_gto(self):
        str1 = b"GET GTO\n"
        self.s.send(str1)
        print("Q : {}".format(str1.decode()), end="")
        data = self.s.recv(1024)
        a = data.decode()
        print("A : {}".format(a))

        flag = int(a[-2:-1])
        if flag == 0:
            print("-->>grp : Gripper is not activated")
            return flag
        elif flag == 1:
            print("-->>grp : Gripper is activated")
            return flag
        else:
            print("!!>>grp : communication error occurred")
            return -1

    def get_sta(self):
        str1 = b"GET STA\n"
        self.s.send(str1)
        print("Q : {}".format(str1.decode()), end="")
        data = self.s.recv(1024)
        a = data.decode()
        print("A : {}".format(a))

        flag = int(a[-2:-1])
        if flag == 0:
            print("-->>grp : Gripper is in reset")
            return flag
        elif flag == 1:
            print("-->>grp : Activation in progress.")
            return flag
        elif flag == 2:
            print("-->>grp : Gripper is not used.")
            return flag
        elif flag == 3:
            print("-->>grp : Activation is completed")
            return flag
        else:
            print("!!>>grp : communication error occurred")
            return -1

    def get_obj(self):
        str1 = b"GET OBJ\n"
        self.s.send(str1)
        print("Q : {}".format(str1.decode()), end="")
        data = self.s.recv(1024)
        a = data.decode()
        print("A : {}".format(a))

        flag = int(a[-2:-1])
        if flag == 0:
            print("-->>grp : Gripper is in reset")
            return flag
        elif flag == 1:
            print("-->>grp : Activation in progress.")
            return flag
        elif flag == 2:
            print("-->>grp : Gripper is not used.")
            return flag
        elif flag == 3:
            print("-->>grp : Activation is completed")
            return flag
        else:
            print("!!>>grp : communication error occurred")
            return -1

