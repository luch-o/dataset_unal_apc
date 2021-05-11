import serial, time

def write_read(x):
    arduino.write(bytes(x, 'utf-8'))
    print("Sending:", x)
    response = arduino.readline().decode('utf-8')
    return response

arduino = serial.Serial('/dev/ttyACM0', 9600, timeout=2)
print("Initializing...")
time.sleep(5)

while True:
    try:
        num = input("Enter an integer: ")
        response = write_read(num)
        time.sleep(0.05)
        print("Receiving:", response)
    except KeyboardInterrupt:
        print("\nTerminating program")
        break
    except Exception as e:
        print("Error:", e)

print("Closing serial port")
arduino.close()