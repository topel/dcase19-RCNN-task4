# -*- coding: utf-8 -*-
"""
created on 23/04/19

@author: Thomas Pellegrini
"""
import os

os.environ["MKL_NUM_THREADS"]="2"
os.environ["NUMEXPR_NUM_THREADS"]="2"
os.environ["OMP_NUM_THREADS"]="2"

import numpy as np
import matplotlib.pyplot as plt
# import matplotlib.cm as cm
# from mpl_toolkits.axes_grid1 import make_axes_locatable
from math import cos, pi

# import soundfile as sf
import librosa
import muda
import jams

import sys

import pickle

import keras

# from keras.models import model_from_json, load_model


from keras import optimizers
from keras.models import Model, Sequential
from keras.layers import Input, Dense, GlobalAveragePooling2D, GlobalMaxPooling2D, Lambda, \
    Flatten, Dropout, Conv2D, MaxPooling2D, BatchNormalization, Activation, TimeDistributed, \
    GRU, Reshape, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D, SpatialDropout2D, \
    Concatenate, Multiply
from keras import backend as K


K.set_image_data_format("channels_last")
K.image_data_format()

K.clear_session()

# fixer les seeds

from numpy.random import seed
seed(123)

from tensorflow import set_random_seed
set_random_seed(1234)


class_correspondance = {"Alarm_bell_ringing": 0, "Speech": 1, "Dog": 2, "Cat": 3, "Vacuum_cleaner": 4,
                        "Dishes": 5, "Frying": 6, "Electric_shaver_toothbrush": 7, "Blender": 8, "Running_water": 9}

class_correspondance_reverse = dict()
for k in class_correspondance:
    class_correspondance_reverse[class_correspondance[k]] = k



def read_csv_list(fpath):
    file_list = []
    with open(fpath, 'rt') as fh:
        for l in fh:
            file_list.append(l.rstrip())

    return file_list


def write_csv_list(lst, fpath):
    with open(fpath, 'wt') as fh:
        for el in lst:
            fh.write('%s\n'%el)

model_index=int(sys.argv[1])
time_dim_out=int(sys.argv[2])
epochs=100

# time_dim_out=431
# time_dim_out=107

from generators import MyGeneratorWeakStrong as MyGenerator

savedir='models_train_synthetic'
#savedir_long='models/%s/model_%d_cosineLR_001_%d'%(savedir, time_dim_out, model_index)
savedir_long='models/%s/model%d_densely'%(savedir, model_index)
# savedir_long='models/%s/model%d_100neurons'%(savedir, model_index)

if not os.path.exists(savedir_long):
    os.makedirs(savedir_long)
print('save dir: %s'%savedir_long)

log_output_fh = open(savedir_long + '/%d.log'%model_index, 'w')
log_output_fh.write('model: %d \n' % (model_index))
log_output_fh.write('savedir: %s \n' % (savedir_long))

# datadir='/baie/travail/thomas/dcase2019/subsets'
datadir='/projets/samova/pellegri/dcase19/subsets'

# wav_lst_train_weak = read_csv_list(datadir + '/' + 'subset_train_weak_1505files_wav.lst')
wav_lst_train_weak_TRAIN = read_csv_list(datadir + '/' + 'subset_train_weak_TRAIN_1354files_wav.lst')
wav_lst_train_weak_DEV = read_csv_list(datadir + '/' + 'subset_train_weak_DEV_151files_wav.lst')
wav_lst_train_synthetic = read_csv_list(datadir + '/' + 'subset_train_synthetic_2045files_wav.lst')
wav_lst_train_uid = read_csv_list(datadir + '/' + 'subset_train_uid_14412files_wav.lst')
wav_lst_train_weak_synthetic = wav_lst_train_weak_TRAIN + wav_lst_train_weak_DEV + wav_lst_train_synthetic

# wav_lst_test_dcase2018 = read_csv_list(datadir + '/' + 'subset_test_dcase2018_923files_wav.lst')
# wav_lst_eval_dcase2018 = read_csv_list(datadir + '/' + 'subset_eval_dcase2018_880files_wav.lst')
# wav_lst_validation = read_csv_list(datadir + '/' + 'subset_validation_1168files_wav.lst')
wav_lst_validation = read_csv_list(datadir + '/' + 'subset_validation_1122files_wav.lst')

# wav_lst_validation = [el for el in wav_lst_validation if el not in validation_to_remove]

# load weak labels dicts
dico_tags_y_train_weak = pickle.load(open(datadir + '/dico_tags_y_train_weak_1505files.pkl', 'rb'))
dico_tags_y_train_synthetic = pickle.load(open(datadir + '/dico_tags_y_train_synthetic_2045files.pkl', 'rb'))
dico_tags_y_train_uid = None
dico_tags_y_validation = pickle.load(open(datadir + '/dico_tags_y_validation_1122files.pkl', 'rb'))

# load strong label dicts
dico_strong_y_train_weak = None
dico_strong_y_train_synthetic = pickle.load(open(datadir + '/dico_strong_y_train_synthetic_2045files.pkl', 'rb'))
dico_strong_y_train_uid = None
dico_strong_y_validation = pickle.load(open(datadir + '/dico_strong_y_validation_1122files.pkl', 'rb'))

# load wav dicts
dico_wav_train_weak = pickle.load(open(datadir + '/dico_wav_train_weak_1505files.pkl', 'rb'))
dico_wav_train_synthetic = pickle.load(open(datadir + '/dico_wav_train_synthetic_2045files.pkl', 'rb'))
dico_wav_validation = pickle.load(open(datadir + '/dico_wav_validation_1122files.pkl', 'rb'))

print(len(dico_tags_y_train_weak), len(dico_tags_y_train_synthetic), len(dico_tags_y_validation))
print(len(dico_strong_y_train_synthetic), len(dico_strong_y_validation))
print(len(dico_wav_train_weak), len(dico_wav_train_synthetic), len(dico_wav_validation))

# merge train_weak and train_synthetic dicts
dico_tags_y_train_weak_synthetic = {**dico_tags_y_train_weak, **dico_tags_y_train_synthetic}
dico_strong_y_train_weak_synthetic = None
dico_wav_train_weak_synthetic = {**dico_wav_train_weak, **dico_wav_train_synthetic}

print(len(dico_tags_y_train_weak_synthetic), len(dico_wav_train_weak_synthetic))

train_batch_size=32
# train_data_folder='/baie/corpus/DCASE2019/task4/dataset/audio/train/weak'

wav_lst=wav_lst_train_weak_synthetic
dico_wav=dico_wav_train_weak_synthetic

lab_dict_weak=dico_tags_y_train_weak_synthetic
lab_dict_strong=dico_strong_y_train_synthetic

train_n_samples=len(wav_lst)
train_fact_amp=0.2
out_dim=10
n_channels=1
n_fbank_coeff=64
time_dim=431
train_shuffle=True
sr=22050 # output by librosa load

nb_steps_train = int(np.ceil(len(wav_lst)/train_batch_size))
print('INFO nb_steps_train:', nb_steps_train)

train_generator = MyGenerator(batch_size=train_batch_size,
                 data_folder=None,
                 wav_lst=wav_lst,
                 n_samples=train_n_samples,
                 dico_wav=dico_wav,
                 lab_dict_weak=lab_dict_weak,
                 lab_dict_strong=lab_dict_strong,
                 sr=sr,
                 fact_amp=train_fact_amp,
                 out_dim=out_dim,
                 perform_data_aug=True,
                 n_channels=n_channels,
                 n_fbank_coeff=n_fbank_coeff,
                 time_dim=time_dim,
                 time_dim_out=time_dim_out,
                 shuffle=train_shuffle)

dev_batch_size=64

wav_lst=wav_lst_validation
dico_wav=dico_wav_validation

lab_dict_weak=dico_tags_y_validation
lab_dict_strong=dico_strong_y_validation

dev_n_samples=len(wav_lst)
dev_fact_amp=0.0
dev_shuffle=False

dev_perform_data_aug=False

nb_steps_dev = int(np.ceil(len(wav_lst)/dev_batch_size))
print('INFO nb_steps_train:', nb_steps_dev)

dev_generator = MyGenerator(batch_size=dev_batch_size,
                 data_folder=None,
                 wav_lst=wav_lst,
                 n_samples=dev_n_samples,
                 dico_wav=dico_wav,
                 lab_dict_weak=lab_dict_weak,
                 lab_dict_strong=lab_dict_strong,
                 sr=sr,
                 fact_amp=dev_fact_amp,
                 out_dim=out_dim,
                 perform_data_aug=dev_perform_data_aug,
                 n_channels=n_channels,
                 n_fbank_coeff=n_fbank_coeff,
                 time_dim=time_dim,
                 time_dim_out=time_dim_out,
                 shuffle=dev_shuffle)


def loc_loss(y_true, y_pred):
    """
    multi-instance loss
    :param y_true: tensor, shape:
    :param y_pred: tensor
    :return: float
    """
    bool_gt = y_true[:, 0, 0]  # a vector of either 0s or 1s
    # bool_gt = bool_gt[:, np.newaxis, :]

    # print('bool_gt', K.int_shape(bool_gt))
    y_t = y_true[:, 1:, :]

    # print('binCE', K.int_shape(K.mean(K.binary_crossentropy(y_t, y_pred), axis=-1)))
    matrix_loss = bool_gt * K.transpose(K.mean(K.binary_crossentropy(y_t, y_pred, from_logits=False), axis=-1))

    # matrix_loss = K.mean(K.binary_crossentropy(y_t, y_pred), axis=-1)
    return K.transpose(matrix_loss)

if time_dim_out < time_dim:
    time_pooling_factor=2
else:
    time_pooling_factor=1


def define_model_densely(at_layer_name='at_output', loc_layer_name='loc_output'):

    time_pooling_factor=1
    
    input_shape = (64, 431, 1)
    
    melInput = Input(input_shape)
    
    # ---- mel convolution part ----
    mBlock1 = Conv2D(filters=64, kernel_size=(3, 3), padding="same")(melInput)
    mBlock1 = BatchNormalization()(mBlock1)
    mBlock1 = Activation(activation="relu")(mBlock1)
    mBlock1 = MaxPooling2D(pool_size=(4, 1))(mBlock1)
    # mBlock1 = Dropout(0.1)(mBlock1)
    mBlock1 = SpatialDropout2D(0.3, data_format=K.image_data_format())(mBlock1)
    
    mBlock2 = Conv2D(filters=64, kernel_size=(3, 3), padding="same")(mBlock1)
    mBlock2 = BatchNormalization()(mBlock2)
    mBlock2 = Activation(activation="relu")(mBlock2)
    mBlock2 = MaxPooling2D(pool_size=(4, time_pooling_factor))(mBlock2)
    mBlock2 = SpatialDropout2D(0.3, data_format=K.image_data_format())(mBlock2)
    # mBlock2 = Dropout(0.1)(mBlock2)
    
    mBlock3 = Concatenate(axis=-1)([MaxPooling2D(pool_size=(4, time_pooling_factor))(mBlock1), mBlock2])
    mBlock3 = Conv2D(filters=64, kernel_size=(3, 3), padding="same")(mBlock3)
    mBlock3 = BatchNormalization()(mBlock3)
    mBlock3 = Activation(activation="relu")(mBlock3)
    mBlock3 = MaxPooling2D(pool_size=(4, time_pooling_factor))(mBlock3)
    mBlock3 = SpatialDropout2D(0.3, data_format=K.image_data_format())(mBlock3)
    # mBlock3 = Dropout(0.1)(mBlock3)
    
    # mBlock4 = mBlock3
    mBlock4 = Concatenate(axis=-1)([MaxPooling2D(pool_size=(4*4, time_pooling_factor))(mBlock1), 
                                    MaxPooling2D(pool_size=(4, time_pooling_factor))(mBlock2), 
                                    mBlock3])
    
    targetShape = int(mBlock4.shape[1] * mBlock4.shape[2])
    mReshape = Reshape(target_shape=(targetShape, int(mBlock4.shape[3])))(mBlock4)
    
    gru = Bidirectional(
        GRU(kernel_initializer='glorot_uniform', activation='tanh', recurrent_dropout=0.1, \
            dropout=0.1, units=64, return_sequences=True)
    )(mReshape)
    
    gru = Dropout(0.1)(gru)
    
    output = TimeDistributed(
        Dense(64, activation="relu"),
    )(gru)
    
    output = Dropout(0.1)(output)
    
    loc_output = TimeDistributed(
        Dense(10, activation="sigmoid"), name=loc_layer_name,
    )(output)
    
    # output = TimeDistributed(
    #  Lambda(lambda x: (x - K.min(x, axis=1, keepdims=True))/(K.max(x, axis=1, keepdims=True)- K.min(x, axis=1, keepdims=True)) ),
    # )(output)
    
    ### output = GlobalAveragePooling1D()(output)
    gap = GlobalAveragePooling1D()(loc_output)
    gmp = GlobalMaxPooling1D()(loc_output)
    # flat_gap = Flatten()(gap)
    # flat_gmp = Flatten()(gmp)
    
    concat = Concatenate()([gap, gmp])
    
    d = Dense(1024, activation="relu")(concat)
    d = Dropout(rate=0.5)(d)
    
    at_output = Dense(10, activation="sigmoid", name=at_layer_name)(d)
    
    model = Model(inputs=[melInput], outputs=[loc_output, at_output])
    return model


def define_model(at_layer_name='at_output', loc_layer_name='loc_output'):

    time_pooling_factor=1
    
    input_shape = (64, 431, 1)

    melInput = Input(input_shape)

    # ---- mel convolution part ----
    mBlock1 = Conv2D(filters=64, kernel_size=(3, 3), padding="same")(melInput)
    mBlock1 = BatchNormalization()(mBlock1)
    mBlock1 = Activation(activation="relu")(mBlock1)
    mBlock1 = MaxPooling2D(pool_size=(4, 1))(mBlock1)
    # mBlock1 = Dropout(0.1)(mBlock1)
    mBlock1 = SpatialDropout2D(0.3, data_format=K.image_data_format())(mBlock1)

    mBlock2 = Conv2D(filters=64, kernel_size=(3, 3), padding="same")(mBlock1)
    mBlock2 = BatchNormalization()(mBlock2)
    mBlock2 = Activation(activation="relu")(mBlock2)
    mBlock2 = MaxPooling2D(pool_size=(4, time_pooling_factor))(mBlock2)
    mBlock2 = SpatialDropout2D(0.3, data_format=K.image_data_format())(mBlock2)
    # mBlock2 = Dropout(0.1)(mBlock2)

    mBlock3 = Conv2D(filters=64, kernel_size=(3, 3), padding="same")(mBlock2)
    mBlock3 = BatchNormalization()(mBlock3)
    mBlock3 = Activation(activation="relu")(mBlock3)
    mBlock3 = MaxPooling2D(pool_size=(4, time_pooling_factor))(mBlock3)
    mBlock3 = SpatialDropout2D(0.3, data_format=K.image_data_format())(mBlock3)
    # mBlock3 = Dropout(0.1)(mBlock3)

    targetShape = int(mBlock3.shape[1] * mBlock3.shape[2])
    mReshape = Reshape(target_shape=(targetShape, 64))(mBlock3)

    gru = Bidirectional(
        GRU(kernel_initializer='glorot_uniform', activation='tanh', recurrent_dropout=0.1, \
            dropout=0.1, units=64, return_sequences=True)
    )(mReshape)

    gru = Dropout(0.1)(gru)

    output = TimeDistributed(
        Dense(64, activation="relu"),
    )(gru)

    output = Dropout(0.1)(output)

    loc_output = TimeDistributed(
        Dense(10, activation="sigmoid"), name=loc_layer_name,
    )(output)

    # output = TimeDistributed(
    #  Lambda(lambda x: (x - K.min(x, axis=1, keepdims=True))/(K.max(x, axis=1, keepdims=True)- K.min(x, axis=1, keepdims=True)) ),
    # )(output)

    ### output = GlobalAveragePooling1D()(output)
    gap = GlobalAveragePooling1D()(loc_output)
    gmp = GlobalMaxPooling1D()(loc_output)
    # flat_gap = Flatten()(gap)
    # flat_gmp = Flatten()(gmp)

    concat = Concatenate()([gap, gmp])

    d = Dense(100, activation="relu")(concat)
    d = Dropout(rate=0.5)(d)

    at_output = Dense(10, activation="sigmoid", name=at_layer_name)(d)

    model = Model(inputs=[melInput], outputs=[loc_output, at_output])
    
    return model
    
model = define_model_densely()

# compile model
# metrics = ["accuracy"]
optimizer = optimizers.Adam(lr=0.001)
# optimizer = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)

# model.compile(loss="binary_crossentropy",
#               optimizer=optimizer)

model.compile(loss={"loc_output": loc_loss, "at_output": "binary_crossentropy"},
              loss_weights={"loc_output": 1., "at_output": 1.},
              optimizer=optimizer)
model.summary(line_length=100)


def lrate_schedule_fn(epoch):
    if epoch < 50:
        return 1e-3
    elif epoch < 100:
        return 5e-4
    elif epoch < 150:
        return 1e-4
    else:
        return 1e-5

# def lrate_schedule_fn(epoch):
#     return 0.001*(1.+cos(epoch*pi/11.))

lrate_scheduler = keras.callbacks.LearningRateScheduler(lrate_schedule_fn)

early_stopper = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=30, verbose=1,
                                              mode='auto')

filepath = os.path.join(savedir_long, "dcase19.{epoch:02d}-{loss:.4f}-{val_loss:.4f}.hdf5")
# create_folder(os.path.dirname(filepath))
model_saver = keras.callbacks.ModelCheckpoint(filepath=filepath,
                                             monitor='val_loss',
                                             verbose=0,
                                             save_best_only=False,
                                             save_weights_only=False,
                                             mode='auto',
                                             period=10)

history = model.fit_generator(
    generator=train_generator,
    epochs=epochs,
    steps_per_epoch=nb_steps_train,
    use_multiprocessing=True,
    validation_data=dev_generator,
    validation_steps=nb_steps_dev,
#     validation_freq=[1, 2, 10, 20, 30, 40, 50],
    workers=os.cpu_count()//2,
    # workers=10,
    callbacks = [lrate_scheduler, early_stopper, model_saver],
    initial_epoch=0,
    verbose=2
    )

for i in range(len(history.history['loss'])):
    log_output_fh.write('loss - ep: %d train: %.3f val:%.3f\n'%(i, history.history['loss'][i], history.history['val_loss'][i]))
with open('%s/model_%d_train_history_dict.pkl'%(savedir_long, model_index), 'wb') as file_pi:
    pickle.dump(history.history, file_pi)

plt.plot(model.history.history['loss'])
plt.plot(model.history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('%s/losses.png'%savedir_long)

model.save(savedir_long + '/model_%d_cnn_%d_epochs_loss_%4f_train_%.4f_val.h5'%(model_index,
                                                                                     epochs,
                                                                                    model.history.history['loss'][-1],
                                                                                    model.history.history['val_loss'][-1]))

val_probs = model.predict_generator(generator=dev_generator,
    steps=nb_steps_dev,
    use_multiprocessing=True,
    workers=os.cpu_count()//2,
    # workers=10,
    max_queue_size=10,
    verbose=2)

val_probs[0] = val_probs[0][:len(wav_lst_validation)]
val_probs[1] = val_probs[1][:len(wav_lst_validation)]


def accuracy_at_from_probs(at_probs, at_gt_dict, file_lst):
    all_y_true = []
    all_y_pred = []
    total_num_true_pos_preds = 0
    total_num_true_pos_gt = 0
    for i, file_id in enumerate(file_lst):
        y_true = at_gt_dict[file_id]
        y_probs = at_probs[i]
        y_preds = 1 * (y_probs > 0.5)
        all_y_true.append(y_true)
        all_y_pred.append(y_preds)
        if i < -1:
            print(i, file_id)
            print(y_true)
            print(y_probs)
            print(y_preds)
        num_true_pos_preds = sum(y_true * y_preds)
        num_true_pos_gt = sum(y_true)
        total_num_true_pos_preds += num_true_pos_preds
        total_num_true_pos_gt += num_true_pos_gt
    if total_num_true_pos_gt > 0:
        return all_y_true, all_y_pred, 100. * total_num_true_pos_preds / total_num_true_pos_gt
    else:
        return all_y_true, all_y_pred, 0.



def score_per_class(all_y_true, all_y_pred):
    from sklearn.metrics import precision_recall_fscore_support
    log_output_fh.write('AT validation')
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    nb_classes = len(all_y_true[0])
    for cl in range(nb_classes):
        class_true = all_y_true[:, cl]
        class_pred = all_y_pred[:, cl]
        p, r, fscore, support = precision_recall_fscore_support(class_true, class_pred, average='binary', pos_label=1)
        log_output_fh.write('%s %.2f %.2f %.2f' % (class_correspondance_reverse[cl], 100 * p, 100 * r, 100 * fscore))


all_y_true, all_y_pred, acc = accuracy_at_from_probs(val_probs[1], dico_tags_y_validation, wav_lst_validation)
score_per_class(all_y_true, all_y_pred)


def rescale(array_1d, border_index=0):
    """rescale between 0 and 1 a 1d-array"""
    #     border_index = 3
    if border_index > 0:
        return (array_1d - np.min(array_1d[border_index:-border_index])) / (
        np.max(array_1d[border_index:-border_index]) - np.min(array_1d[border_index:-border_index]))
    else:
        return (array_1d - np.min(array_1d)) / (np.max(array_1d) - np.min(array_1d))


def get_prob_curves_for_predicted_classes(audio_tag_probs, strong_probs, featTestList, int2className):
    audio_tag_preds = 1 * (audio_tag_probs > 0.5)

    dico_prob_curves_for_predicted_classes = {}
    for i, fileid in enumerate(featTestList):
        dico_prob_curves_for_predicted_classes[fileid] = {}
        current_classes = np.nonzero(audio_tag_preds[i])[0]
        for j in range(current_classes.shape[0]):
            class_name = int2className[current_classes[j]]
            dico_prob_curves_for_predicted_classes[fileid][class_name] = rescale(strong_probs[i, :, current_classes[j]],
                                                                                 border_index=0)
            #             dico_prob_curves_for_predicted_classes[fileid_short][class_name] = strong_probs[i,:,current_classes[j]]

    return dico_prob_curves_for_predicted_classes


dico_prob_curves = get_prob_curves_for_predicted_classes(val_probs[1],
                                                         val_probs[0],
                                                         wav_lst_validation,
                                                         class_correspondance_reverse)


def from_strong_probs_dict_2_strong_probs_3d_array(dico_prob_curves_for_predicted_classes, wav_lst_validation):
    nbFrame = 431
    out_dim = 10

    # rearange list of results
    results = np.zeros((len(wav_lst_validation), nbFrame, out_dim))

    for i, f in enumerate(wav_lst_validation):
        dico_strong_probs = dico_prob_curves_for_predicted_classes[f]
        curves = np.zeros((nbFrame, out_dim))

        for event, prob in dico_strong_probs.items():
            curves[:, class_correspondance[event]] = prob

        results[i] = curves
    return results


strong_probs_3d_array = from_strong_probs_dict_2_strong_probs_3d_array(dico_prob_curves, wav_lst_validation)


def load_test_strong_ground_truth(fpath):
    # test ground truth
    with open(fpath, "r") as f:
        str_strong_y_true = f.read().splitlines()[1:]

    to_be_removed = []
    for el in str_strong_y_true:
        info = el.split("\t")
        if info[1] == '':
            to_be_removed.append(el)
    return [el for el in str_strong_y_true if el not in to_be_removed]


gt_fpath = datadir + '/validation.csv'
str_strong_y_true = load_test_strong_ground_truth(gt_fpath)


from sed_tool.optimizers import DichotomicOptimizer
from sed_tool.Encoder import Encoder
# from sed_tool.sed_tools import event_based_evaluation
from sed_tool.sed_tools import eb_evaluator, sb_evaluator

import dcase_util as dcu


# Find best hysteresis threshold parameters
def eval_hysteresis(test_strong_y_true, time_prediction, name_list, low=(0.001, 0.09), high=(0.001, 0.09),
                    step=5, recurse=10):
    """Find best hysteresis threshold parameters.

    :param test_strong_y_true: GT strong annotations
    :param time_prediction: The time prediction that will be converted into segments and evaluate
    :param name_list: The list of file names (must be in the same order than time_prediction)
    :param low: The search boundaries for the "low" parameter (low -> bottom threshold)
    :param high: the search boundaries for the "high" parameter (high -> top threshold)
    :param nb_digit: The threshold max precision
    :param step: The number of value that will be test in between the boudaries (for each parameters)
    :param recurse: The number of time the search will be done (each time in between the two best boundaries of the previous execution)
    :param monitor: The metrics to focus on for comparing the results of each parameters combination.
    """

    class_list = ['Alarm_bell_ringing', 'Speech', 'Dog', 'Cat', 'Vacuum_cleaner', 'Dishes', 'Frying',
                  'Electric_shaver_toothbrush', 'Blender', 'Running_water']

    # Create the encoder that will be used
    encoder = Encoder(
        classes=class_list,
        temporal_precision=200,  # ms
        clip_length=10,  # s
        minimal_segment_step=200  # ms
    )

    optimizer = DichotomicOptimizer(
        {
            "low": low,
            "high": high,
            "smooth": "smoothMovingAvg",
            "window_len": (5, 27)  # [19]
        },

        encoder=encoder,
        step=step,
        nb_recurse=recurse,
        nb_process=8
    )

    # Perform the optimization ON TEST
    optimizer.fit(
        y_true=test_strong_y_true,
        y_pred=time_prediction,
        filenames=name_list,
        method="hysteresis",
        verbose=2
    )

    return optimizer


def apply_hysteresis(best_params, str_strong_y_true, strong_prediction, filenames):

    class_list = ['Alarm_bell_ringing', 'Speech', 'Dog', 'Cat', 'Vacuum_cleaner', 'Dishes', 'Frying', 'Electric_shaver_toothbrush', 'Blender', 'Running_water']

    # Create the encoder that will be used
    encoder = Encoder(
        classes=class_list,
        temporal_precision = 200,  # ms
        clip_length = 10,          # s
        minimal_segment_step = 200 # ms
    )

    segments = encoder.encode(
        strong_prediction,
        method="hysteresis",
        **best_params
    )

    to_evaluate = encoder.parse(segments, filenames)
#     print(type(to_evaluate))
    event_results = eb_evaluator(str_strong_y_true, to_evaluate)
    log_output_fh.write(event_results)

    sb_results = sb_evaluator(str_strong_y_true, to_evaluate, time_resolution=1.)
    log_output_fh.write(sb_results)

best_params = {'low': 0.1, 'high': 0.3, 'window_len': 10., 'smooth': 'smoothMovingAvg'}
apply_hysteresis(best_params, str_strong_y_true, strong_probs_3d_array, wav_lst_validation)






