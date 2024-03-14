
# run the project with the default image
run-default:
  board-recognition -i data/30.jpg

# display usage
usage:
  @just --list

# run the project
run *ARGS:
  board-recognition {{ ARGS }}

# watch the project
watch *ARGS:
  watchexec -r --exts py board-recognition {{ ARGS }}

# run the project with the default image
watch-default:
  watchexec -r --exts py board-recognition -i data/30.jpg

# measure the performance of the project
measure *ARGS:
  board-recognition-measure {{ ARGS }}

# watch the measure of performance of the project
watch-measure:
  watchexec -r --exts py board-recognition-measure

# formats all files
fmt:
  poetry run black .
  poetry run isort .

# tests the project
test:
  poetry run pytest

# tests the project on file change
watch-test:
  watchexec --exts py poetry run pytest
