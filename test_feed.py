import myo as libmyo; libmyo.init(r'C:\Users\Owner\Desktop\BIOEN 461\myo-sdk-win-0.9.0\myo-sdk-win-0.9.0\bin')
import time
import sys
import numpy as np

feed = libmyo.device_listener.Feed()
hub = libmyo.Hub()
hub.run(1000, feed)

def send_vibe(param):
    myo.vibrate("short")
    #linear
    #delay = param * (1.2/6.5) + 0.3
    delay = 0.3 * np.exp(param)
    time.sleep(delay)

def angle_shift(angles, f, l, r)
    angles = angles - f


try:
    myo = feed.wait_for_single_device(timeout=5.0)  # seconds
    if not myo:
        print("No Myo connected after 5 seconds.")
        sys.exit()

    # initialization
    if hub.running and myo.connected:
        print('Get Ready')
        time.sleep(3)
        print('Hold arm straight')
        time.sleep(3)
        baseline = myo.orientation
        baseline_rpy = np.array(baseline.rpy)
        print(baseline_rpy)
        print()
        print('Hold arm left')
        time.sleep(3)
        left_rpy = np.array(myo.orientation.rpy)
        print(left_rpy)
        print()
        print('Hold arm right')
        time.sleep(3)
        right_rpy = np.array(myo.orientation.rpy)
        print(right_rpy)
        print()

    while hub.running and myo.connected:
        quat = myo.orientation
        rpy = np.array(quat.rpy)
        diff = baseline_rpy[1:2] - rpy[1:2]
        diff_mag = np.square(diff)
        diff_mag = np.sum(diff_mag)
        diff_mag = np.sqrt(diff_mag)
        print(diff_mag)
        send_vibe(diff_mag)

except KeyboardInterrupt:
    print("Quitting...")
finally:
    hub.shutdown()
