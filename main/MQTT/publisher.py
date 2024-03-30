import paho.mqtt.client as mqtt
import time

# in main event
MQTT_BROKER_HOST = 'broker.hivemq.com'
MQTT_BROKER_PORT = 1883
MQTT_KEEP_ALIVE_INTERVAL = 60

client = mqtt.Client()
client.connect(MQTT_BROKER_HOST, MQTT_BROKER_PORT, MQTT_KEEP_ALIVE_INTERVAL)

client.publish("raspberry/speaker", "sentence to say");

time.sleep(1)

client.publish("raspberry/speaker","sentence 2")

time.sleep(1)

client.publish("raspberry/speaker","sentence 3")
