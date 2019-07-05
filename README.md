# Entry for the DCASE 2019 Task 4 challenge: Sound event detection in domestic environments

Submission name: ``PELLEGRINI_IRIT_task4_1"

Authors: 

- Thomas Pellegrini, thomas.pellegrini@irit.fr
- Leo Cances, leo.cances@irit.fr

The official results can be found in <http://dcase.community/challenge2019/task-sound-event-detection-in-domestic-environments-results>

Event-based F-score:

- Eval dataset: 39.7% 
- Development dataset ("validation" subset): 39.9%

The repository contains:
- a complete notebook to train or load a model, optimize thresholds, evaluate on the validation subset and generate a submission: [dcase19_PELLEGRINI_IRIT_task4_1.ipynb](dcase19_PELLEGRINI_IRIT_task4_1.ipynb)
- the original python script to train a model: [dcase19.py](dcase19.py)
- the model (weights, hdf5 format) used to obtain our official submission ``PELLEGRINI_IRIT_task4_1": [dcase19.90-0.1658-0.3292.h5](dcase19.90-0.1658-0.3292.h5)
- the thresholds for audio tagging optimized on the val subset for this model: [AT_thresholds.pkl](AT_thresholds.pkl)
- the thresholds for event localization optimized on the val subset for this model: [HYSTERESIS_params.pkl](HYSTERESIS_params.pkl)

To ease reproductibilty, we also share our input data files (file id lists and waveform dictionaries): [link](https://drive.google.com/drive/folders/1EnNmihEJXe8JlUFxLv1c9tDOUPgtH8OC?usp=sharing).

The main contribution of the present work lies in the threshold optimization routines that we compiled in a toolbox still in development: sed_tool

It can be installed this way: 

    pip install -i https://test.pypi.org/simple sed_tool

The submission to the challenge was made with a single small RCNN model (about 165k params).

![model Image](https://github.com/topel/dcase19-RCNN-task4/blob/master/model_1.png)

- "at": Audio Tagging, "loc": localization

The model has been trained on the weak and synthetic training datasets with binary cross-entropy, respectively at recording-level for both subsets and at frame-level for the synthetic subset that provides strong labels. We originally trained it for 120 epochs but the best model on the val subset was the one obtained after 90 epochs only.

Class-dependent thresholds are optimized on the validation subset for:

- audio tagging with simple hard thresholds,
- event localization using a hysteresis thresholding method (a high and a low thresholds for each class).

We implemented other thresholding methods for localization available in sed_tool, namely "absolute" and "slope-based" thresholding methods. Hysteresis gave the best results. For more details on this, please refer to [2].

Large performance gains were obtained thank to the two threshold optimizations showing how it is crucial setting appropriate thresholds for each class.

There is still a large room for improvement, in particular regarding the audio tagging capability of this small model. Furthermore, the unlabeled in-domain subset was not used.

If you use this code, please consider citing:

[1] Leo Cances, Patrice Guyot, Thomas Pellegrini. Multi-task learning and post processing optimization for sound event detection. Technical Report, DCASE 2019, http://dcase.community/documents/challenge2019/technical_reports/DCASE2019_Cances_69.pdf

[2] Leo Cances, Patrice Guyot, Thomas Pellegrini. Evaluation of post-processing algorithms for polyphonic sound event detection. [arXiv:1906.06909](https://arxiv.org/abs/1906.06909)
