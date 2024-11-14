from csi_file_opener import csi_file_process
from multiprocessing import Pool, cpu_count
from glob import glob
import pandas as pd
from tqdm import tqdm


class BaseData(object):
    def set_num_processes(self, n_proc):
        if (n_proc is None) or (n_proc <= 0):
            self.n_proc = cpu_count()  # max(1, cpu_count() - 1)
        else:
            self.n_proc = min(n_proc, cpu_count())


class CsiData(BaseData):
    def __init__(self, dir="../data/widar_dataset/CSI", n_proc=1):
        super().__init__()
        self.set_num_processes(n_proc=n_proc)

        self.gesture_label_numbering = pd.read_csv(dir+'/widar_data_label.csv', index_col=['date', 'user'])
        self.gestures_true_id = {str(name) : idx for idx, name in enumerate(self.gesture_label_numbering.columns)}

        self.data_df = self.load_all(dir)


    def load_all(self, dir="../data/widar_dataset/CSI"):
        data_list = []

        dir_list = glob(dir+"/**/*.dat", recursive=True)

        _n_proc = min(self.n_proc, len(dir_list))  # no more than file_names needed here

        input_list = [(dir, self.gesture_label_numbering, self.gestures_true_id) for dir in dir_list]

        if _n_proc > 1:
            with Pool(processes=_n_proc) as pool:
                data_list = list(pool.map(CsiData.process_single_file_star, input_list))
                # data_list.append(data_series)
        else:
            for input in input_list:
                data_series = CsiData.process_single_file(input)
                data_list.append(data_series)

        return pd.DataFrame(data_list)
    
    @staticmethod
    def process_single_file_star(args):
        return CsiData.process_single_file(*args)

    @staticmethod
    def process_single_file(dir, gesture_label_numbering, gestures_true_id):
        data_backbone = csi_file_process(dir)

        def gesture_name_parser(gesture_label_numbering : pd.DataFrame, gestures_true_id, date, user, g_id):
            gesture_n = str(gesture_label_numbering.columns[gesture_label_numbering.loc[date, user]== g_id].values[0])
            num = gestures_true_id[gesture_n]
            return num


        data_backbone["gesture"] = gesture_name_parser(gesture_label_numbering,
                                                       gestures_true_id,
                                                       data_backbone["date"],
                                                       data_backbone["user"],
                                                       data_backbone["gesture"])
        
        return data_backbone
        

if __name__=="__main__":
    my_data = CsiData(n_proc=96)

    my_data.data_df.to_csv("../data/result.zip", compression="zip")

