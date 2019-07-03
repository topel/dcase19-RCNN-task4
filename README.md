# dcase19-RCNN-task4

Entry for the DCASE 2019 Task 4 challenge: Sound event detection in domestic environments

Submission name: ``PELLEGRINI_IRIT_task4_1"

Authors: 

- Thomas Pellegrini, thomas.pellegrini@irit.fr
- Leo Cances, leo.cances@irit.fr

The official results can be found in <http://dcase.community/challenge2019/task-sound-event-detection-in-domestic-environments-results>

Event-based F-score:

- Eval dataset: 39.7% 
- Development dataset: 39.9% (40.9% in this notebook)

The repository contains:
- a complete notebook to train or load our model, evaluate it on the validation subset and generate a submission,
- the original python script to train a model
- the model used to obtain our official submission ``PELLEGRINI_IRIT_task4_1"
- the thresholds for audio tagging optimized on the val subset
- the thresholds for event localization optimized on the val subset

The main contribution of the present work lies in the threshold optimization routines that we compiled in a toolbox still in development: sed_tool

It can be installed this way: 
    pip install -i https://test.pypi.org/simple sed_tool

The submission to the challenge was made with a single small RCNN model (about 165k params).

![model Image](https://github.com/topel/dcase19-RCNN-task4/blob/master/model_1.png)

The model has been trained on the weak and synthetic training datasets with binary cross-entropy, respectively at recording-level for both subsets and at frame-level for the synthetic subset that provides strong labels. We originally trained it for 120 epochs but the best model on the val subset was the one obtained after 90 epochs only.


Class-dependent thresholds are optimized on the validation for:

- audio tagging with a genetic-like algorithm,
- event localization using a hysteresis thresholding method (a high and a low thresholds for each class).

Large performance gains were obtained thank to the two threshold optimizations showing how it is crucial setting appropriate thresholds for each class.

There is still a large room for improvement, in particular regarding the audio tagging capability of this small model. Furthermore, the unlabeled in-domain subset was not used.

If you use this code, please consider citing:

Leo Cances, Patrice Guyot, Thomas Pellegrini. Multi-task learning and post processing optimization for sound event detection. Technical Report, DCASE 2019, http://dcase.community/documents/challenge2019/technical_reports/DCASE2019_Cances_69.pdf

Leo Cances, Patrice Guyot, Thomas Pellegrini. Evaluation of post-processing algorithms for polyphonic sound event detection. arXiv:1906.06909
