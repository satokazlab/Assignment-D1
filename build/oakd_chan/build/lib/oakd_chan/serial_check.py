import depthai as dai

# 接続されているすべてのデバイスを取得
available_devices = dai.Device.getAllAvailableDevices()

if not available_devices:
    print("No devices found!")
    exit()

print("Available devices:")
for device in available_devices:
    print(f"Serial Number (MxId): {device.getMxId()}")
