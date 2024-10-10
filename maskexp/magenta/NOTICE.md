### Modifications Notice
This repository contains code from the Magenta project, originally licensed under the Apache License 2.0. Modifications have been made to the original source code, primarily to adjust import statements for compatibility with TensorFlow.

E.g.
```python
from tensorflow import compat as ttf
tf = ttf.v1
```