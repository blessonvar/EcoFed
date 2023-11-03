# Communicator Object
import pickle
import struct
import socket
import config

import logging
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Communicator(object):
	def __init__(self):
		self.sock = socket.socket()
		self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
		self.comm_cost = 0


	def send_msg(self, sock, msg):
		msg_pickle = pickle.dumps(msg)
		self.comm_cost += (len(msg_pickle)/(1024*1024))
		sock.sendall(struct.pack(">I", len(msg_pickle)))
		sock.sendall(msg_pickle)
		logger.debug(msg[0]+'sent to'+str(sock.getpeername()[0])+':'+str(sock.getpeername()[1]))

	def recv_msg(self, sock, expect_msg_type=None):
		msg_len = struct.unpack(">I", sock.recv(4))[0]
		msg = sock.recv(msg_len, socket.MSG_WAITALL)
		self.comm_cost += (len(msg)/(1024*1024))
		msg = pickle.loads(msg)
		logger.debug(msg[0]+'received from'+str(sock.getpeername()[0])+':'+str(sock.getpeername()[1]))

		if expect_msg_type is not None:
			if msg[0] == 'Finish':
				return msg
			elif msg[0] != expect_msg_type:
				raise Exception("Expected " + expect_msg_type + " but received " + msg[0])
		return msg