import paho.mqtt.client as mqtt
# in // event
# import pyttsx3

MQTT_BROKER_HOST = 'broker.hivemq.com'
MQTT_BROKER_PORT = 1883
MQTT_KEEP_ALIVE_INTERVAL = 60

# for e speak python
# engine = pyttsx3.init()

def on_connect(client, userdata, flags, rc):
    print("Connected with result code " + str(rc))
    client.subscribe("raspberry/speaker")
def on_message(client, userdata, msg):
    print("Message Recieved. :- ", msg.payload.decode())
    message = msg.payload.decode()
    # speak(message)...
    # engine.say(message)
    # engine.runAndWait()


client = mqtt.Client()
client.connect(MQTT_BROKER_HOST, MQTT_BROKER_PORT, MQTT_KEEP_ALIVE_INTERVAL)
client.on_connect = on_connect
client.on_message = on_message
client.loop_forever()
