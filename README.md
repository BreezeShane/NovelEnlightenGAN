# Novel Enlighten GAN

To use the project, please install the packages.
```yaml
torch~=1.9.0
numpy~=1.19.5
torchvision~=0.10.0
pillow~=8.2.0
opencv-python~=4.5.3.56
flask~=1.1.2
flask_uploads~=0.2.1
argparse~=1.4.0
```
And then you need to edit the package file: `{Path_to_your_environment}/lib/python3.8/site-packages/flask_uploads.py`
Change
```python
from werkzeug import secure_filename,FileStorage
```
to
```python
from werkzeug.utils import secure_filename
from werkzeug.datastructures import  FileStorage
```
Last, you can run the command to use the app:
```shell
python Web.py
```
If you want to run the experiment version, you need to run:
```shell
python Main.py --train
```
to train, and you can also run:
```shell
python Main.py --predict
```
to predict.
Of course we prepared the mode named "continue_train", to use it, you can run: 
```shell
python Main.py --continue_train.
```
For more attributes you can edit, please read [Config.py](./Config.py) to choose those you want to edit!

---

## Where is it from ?

It comes from [here](https://github.com/VITA-Group/EnlightenGAN). And the original belongs to the team [VITA Group](https://github.com/VITA-Group). I just rebuild the project and am attempting to propose a better method to improve it !
