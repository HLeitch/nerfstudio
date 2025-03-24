import os
import subprocess
import sys


def recursiveSearch(directory, videoList):
    print(f"Directory Searching = {directory}")
    filesInDir = os.listdir(directory)
    for file in filesInDir:

        fullLocation = directory + "\\" + file
        if file[-3:] == "mp4":
            print(f"mp4 file found: {file}")
            videoList.append(fullLocation)

        if "." not in file:
            continue
            ##recursiveSearch(fullLocation, videoList)
    return videoList


if __name__ == "__main__":
    print("Which folder to recurse through?: ")
    root_folder = input()
    print("Which NeRF method?: ")
    method = input()
    print("Identifier:")
    ident = input()

    videoFiles = list()

    videoFiles = recursiveSearch(root_folder, videoFiles)

    for video in videoFiles:
        print(video)

        ##reduce to directory
        directory = os.path.dirname(video)
        if "colmap" not in os.listdir(directory):
            os.system(
                f"ns-process-data video --data {video} --output-dir {directory} --matching-method sequential --verbose --num-frames-target 300 "
            )

    for video in videoFiles:
        directory = os.path.dirname(video)
        timestamp = method + os.path.basename(video)[0:-4]+ ident

        os.curdir = directory
        print(f"Training Start: {directory}")
        os.system(
            f"ns-train {method} --data {directory} --output-dir .\\ --timestamp {timestamp} --viewer.quit-on-train-completion True --trainer.max-num-iterations 10000 --vis tensorboard"
        )

        print("Training Finished")
        os.system(f"ns-eval --load-config {directory}\\{method}\\{timestamp}\\config.yml --output-path {directory}\\{timestamp}.json")
        print("Evaluation Finished")
        os.system(
            f"ns-render --load-config {directory}\\{method}\\{timestamp}\\config.yml --traj filename --camera-path-filename C:\\Users\\hleit\\Documents\\nerfstudio\\CamPath\\DecayCamPath.json --seconds 15 --output-path {directory}\\{method}\\{timestamp}.mp4"
        )
