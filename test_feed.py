import myo as libmyo; libmyo.init(r'C:\Users\Owner\Desktop\BIOEN 461\myo-sdk-win-0.9.0\myo-sdk-win-0.9.0\bin')
import time
import sys
import numpy as np

feed = libmyo.device_listener.Feed()
hub = libmyo.Hub()
hub.run(1000, feed)
try:
    myo = feed.wait_for_single_device(timeout=5.0)  # seconds
    if not myo:
        print("No Myo connected after 5 seconds.")
        sys.exit()

    # initialization
    if hub.running and myo.connected:
        print('Collecting initial orientation...')
        print('Hold still')
        time.sleep(2)
        baseline = myo.orientation
        baseline_rpy = np.array(baseline.rpy)
        print(baseline_rpy)
        print()

    while hub.running and myo.connected:
        quat = myo.orientation
        rpy = np.array(quat.rpy)
        diff = baseline_rpy - rpy
        print(diff)
        time.sleep(0.5)

except KeyboardInterrupt:
    print("Quitting...")
finally:
    hub.shutdown()
