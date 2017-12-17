The programs require:
- python 3
- tensorflow 1.0.0 (the specific version is important)
- spacy
- nltk


To run training or testing of corpuses, change the mode on line 69 of the Facebot files and run:

- python3 Facebot.py

for corpus 1 and

- python3 Facebot2.py

for corpus 2.

To run the Chatbot gui on corpus 2*, run:

- python3 gui.py

*(This requires having a trained model. Since the model checkpoint files were too big we didn't include them.)


File structure:

The datafiles and scripts for processing data are in the datasets folder. This also contains the test results.

The saved models are in the ckpt folder.
