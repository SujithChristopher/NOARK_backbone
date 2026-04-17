import serial

def main():
    ser = serial.Serial('/dev/ttyACM0', 15200)  # Update with your serial port and baud rate
    while True:
        if ser.in_waiting > 0:
            line = ser.readline().decode('utf-8').rstrip()
            print(line)
if __name__ == "__main__":
    main()