import socketio

io = socketio.server()
app = socketio.WSGIApp(sio)

