import sys,os
sys.path.append("pydevd-pycharm.egg")
import pydevd_pycharm

pydevd_pycharm.settrace('103.46.128.41', port=15512, stdoutToServer=True,
                        stderrToServer=True)
sys.path.append(os.pardir)


from lane_change_risk_detection.dataset import *
from lane_change_risk_detection.models import Models
from keras.applications import ResNet50
from keras.models import Model


class_weight = {0: 0.05, 1: 0.95}
training_to_all_data_ratio = 0.9
nb_cross_val = 1
nb_epoch = 1000
batch_size = 32

dir_name = os.path.dirname(__file__)
dir_name = os.path.dirname(dir_name)

#image_path = os.path.join(dir_name, 'data/input')
image_path = os.path.join(os.pardir, 'data/input')

backbone_model = ResNet50(weights='imagenet')
backbone_model = Model(inputs=backbone_model.input, outputs=backbone_model.get_layer(index=-2).output)

data = DataSet()
data.model = backbone_model
data.extract_features(image_path, option='fixed frame amount', number_of_frames=50)  #load image as a sequence
data.read_risk_data("LCTable.csv")
data.convert_risk_to_one_hot(risk_threshold=0.1)    #original value=0.05

Data = data.video_features
#Data=Data.reshape(Data.shape[0]*Data.shape[1],1,Data.shape[2]) #make images in all folders in one dimensional
#Data=data.video_features.transpose((1,0,2))  #Data=Data.T
label = data.risk_one_hot

model = Models(nb_epoch=nb_epoch, batch_size=batch_size, class_weights=class_weight)
model.build_transfer_LSTM_model(input_shape=Data.shape[1:])   # resnet features-> LSTM network-> softmax output
model.train_n_fold_cross_val(Data, label, training_to_all_data_ratio=training_to_all_data_ratio, n=nb_cross_val, print_option=0, plot_option=0, save_option=0)
model.model.save('new_resnet_lstm.h5')