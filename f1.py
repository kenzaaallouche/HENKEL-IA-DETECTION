from roboflow import Roboflow
rf = Roboflow(api_key="KFRU55MFH8DYV5mu1EDh")
project = rf.workspace("bingovissel-fcfyn").project("liquide-vaisselle")
version = project.version(1)
dataset = version.download("yolov8")
