import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/suguta/naka2_ws/src/oakd_chan/install/oakd_chan'
