import os
from shutil import copyfile

actor_folders = os.listdir("data/")

root_folder = 'emotions/'
emotions = ['neutral','calm','happy','sad','angry','fearful','disgust','surprised']

if not os.path.exists(root_folder):
    os.makedirs(root_folder)

for emotion in emotions:
    if not os.path.exists(root_folder+emotion):
        os.makedirs(root_folder+emotion)

for dname in actor_folders:
    for fname in os.listdir(dname):
        class_emotion = fname[7]
        class_int = int(class_emotion) - 1
        print(dname,fname,class_emotion, class_int, int(class_emotion))
        src = 'data/'+dname+'/'+fname
        dst = root_folder+emotions[class_int]+'/'+fname
        copyfile(src, dst)
