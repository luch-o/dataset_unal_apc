import serial, time

class Turntable:
    def __init__(self, port: str = '/dev/ttyACM0', baudrate: int = 9600, timeout: float = 2) -> serial.Serial:
        """
        Initialize turntable serial port, returns serial.Serial object representing
        the turntable.
        port, baudrate and timeout are set by default but can be specified.
        """
        self.turntable = serial.Serial(port=port, baudrate=baudrate, timeout=timeout)
        print("Initializing...")
        time.sleep(3)
        print("Done")

    def move_table(self, steps: int, verbose: bool = False, prompt: str = '>'):
        """
        Send an integer number to arduino controlling the turntable.
        If verbose is set to true, mesagges configured on arduino program
        will be printed. Angle must be an integer
        """
        
        self.turntable.write(bytes(str(steps), 'utf-8'))
        
        for _ in range(2):
            msg = self.turntable.readline().decode('utf-8').rstrip()
            if verbose: print(prompt, msg)

        # wait for confirmation
        print("Waiting confirmation")
        while True:
            response = self.turntable.readline()
            if response:
                print(prompt, response.decode('utf-8').rstrip())
                break

    def query_position(self, prompt: str = '>'):
        """
        Query current turntable position
        """
        self.turntable.write(b'0')
        print(prompt, self.turntable.readline().decode('utf-8').rstrip())

    def close(self):
        self.turntable.close()

if __name__ == '__main__':
    turntable = Turntable()
    while True:
        try:
            steps = input("Enter a number of steps: ")
            if steps == '0':
                turntable.query_position()
            else:
                turntable.move_table(steps, verbose=True)
        except KeyboardInterrupt:
            print("\nTerminating program")
            break
        except Exception as e:
            print("Error:", e)

    print("Closing serial port")
    turntable.close()
