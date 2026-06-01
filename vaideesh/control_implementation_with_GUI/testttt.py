import serial.tools.list_ports

ports = serial.tools.list_ports.comports()
print("\n[available ports]")
for i, p in enumerate(ports):
    print(f"  [{i}] {p}")
print()