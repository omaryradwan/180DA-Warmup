import paho.mqtt.client as mqtt

def on_connect(client,userdata,flags,rc):
        print("Connected Returned Result: " + str(rc));
        client.subscribe("omar/radwan",qos=1);

def on_disconnect(client,userdata,rc):
        if rc != 0:
                print("Unexpected Disconnect");
        else:
                print("Expected Disconnect");
def on_message(client, userdata,message):
        print("Receive Message: " + str(message.payload) + " on topic " + message.topic + " with QoS " + str(message.qos));
client = mqtt.Client();
client.on_connect = on_connect;
client.on_disconnect = on_disconnect;
client.on_message = on_message;

client.connect_async("mqtt.eclipse.org");
client.loop_start();

while True:
        pass;
client.loop_stop();
client.disconnect();
