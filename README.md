# Board recognition

This project is done during a class at Université Paris Cité.
The objective is to use image processing techniques to infer information about a whiteboard or blackboard.

For this project, we have three goals :

- detecting the edges of the board
- detecting the text areas on the board
- detecting schemas areas on the board

## Installation

To install and run this project, you need to install poetry:

```
pip install poetry
```

Then you need to get the directory on your computer:

```
git clone ssh@github.com:paulchambaz/board-recognition.git
cd board-recognition
```

Then to get in the virtual python environment, you need to start the poetry shell:

```
poetry shell
poetry install
board-recognition --image data/0.jpg
```

## Group members

This project was done by David Bret, Paul Chambaz and Frédéric Ye
