from tensorboard import program
import os

#path to the log files saved location
__file__ = os.path.abspath('')

location = 'your_path_to_folder_of_log'

#if __name__ == "__main__":
tb = program.TensorBoard()
tb.configure(argv=[None, '--logdir', location])
url = tb.launch()
#print(f"Tensorflow listening on {url}")
