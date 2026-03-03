import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
from camera import MainClass
from pyteensy import TeensyPort,EncoderProcessor


class XYPlot:
    def __init__(self):
        calib_path = '/home/sujith/Documents/rpi_python/old_calibration/calib_mono_faith3D.toml'
        self.cam = MainClass(calib_path)
        self.enc = TeensyPort()
        self.enc.start()
        self.enc_value = EncoderProcessor(self.enc)



    def run(self):
        Figure,axes = plt.subplot(2,2)
        # XY plot
        axes[0,0].plot(x,y,'ro',markersize = 8)
        axes[0,0].set_xlabel("X")
        axes[0,0].set_ylabel("Y")
        axes[0,0].set_title("XY Plot")
        axes[0,0].set_xlim(-50,50)
        axes[0,0].set_ylim(-40,40)

        # Encoder 1
        axes[0,1].plot([],[],'b-',lw =2)
        axes[0,1].set_xlabel("Time")
        axes[0,1].set_ylabel("Encoder Value")
        axes[0,1].set_title("Encoder 1")
        x_data_1 = []
        y_data_1 = []
        # Encoder 2
        axes[1,0].plot([],[],'g-',lw = 2)
        axes[1,0].set_xlabel("Time")
        axes[1,0].set_ylabel("Encoder Value")
        axes[1,0].set_title("Encoder 2")
        x_data_2 = []
        y_data_2 = []
        
        while True:
            #XY Data
            self.cam.process_frame()
            tvec = self.cam.final_pos_cm
            if tvec is None or len(tvec) < 3:
                # Return dummy values if no marker was found in the frame
                return 0.0, 0.0 
            x = round(tvec[0])          # raw camera X
            y = round(tvec[2])          # raw camera Z used as Y
            axes[0,0].set_data(x,y)
            #   Encoder 1 and 2 data
            enc_value_1 = float(self.enc_value.getvalue_1())
            x_data_1.append()
            y_data_1.append(enc_value_1)
            axes[0,1].set_data(x_data_1,x_data_2)

            enc_value_2 = float(self.enc_value.getvalue_2())
            x_data_2.append()
            y_data_2.append(enc_value_2)
            axes[1,0].set_data(x_data_2,y_data_2)
            print("E1:,E2:",enc_value_1,enc_value_2)

            plt.tight_layout()
            plt.show()



if __name__ == "__main__":
    graph = XYPlot()
    try:
        graph.run()
    except KeyboardInterrupt:
        print("\nExiting...")
        