import serial, time

def initialize(port: str = '/dev/ttyACM0', baudrate: int = 9600, timeout: float = 2) -> serial.Serial:
    """
    Initialize turntable serial port, returns serial.Serial object representing
    the turntable.
    port, baudrate and timeout are set by default but can be specified.
    """
    turntable = serial.Serial(port=port, baudrate=baudrate, timeout=timeout)
    print("Initializing...")
    time.sleep(3)
    print("Done")
    return turntable

def move_table(angle: int, verbose: bool = False, prompt: str = '>'):
    """
    Send an integer number to arduino controlling the turntable.
    If verbose is set to true, mesagges configured on arduino program
    will be printed. Angle must be an integer
    """
    
    turntable.write(bytes(str(angle), 'utf-8'))
    
    if verbose:
        for _ in range(3):
            print(prompt, turntable.readline().decode('utf-8').rstrip())

    # wait for confirmation
    while True:
        response = turntable.readline()
        if response:
            print(prompt, response.decode('utf-8').rstrip())
            break

def query_position(prompt: str = '>'):
    """
    Query current turntable position
    """
    turntable.write(b'0')
    print(prompt, turntable.readline().decode('utf-8').rstrip())

if __name__ == '__main__':
    turntable = initialize()
    while True:
        try:
            angle = input("Enter an integer: ")
            if angle == '0':
                query_position()
            else:
                move_table(angle, verbose=True)
        except KeyboardInterrupt:
            print("\nTerminating program")
            break
        except Exception as e:
            print("Error:", e)

    print("Closing serial port")
    turntable.close()
