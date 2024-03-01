В терминале запустить команду
python faceforensics_download_v4.py -d DeepFakeDetection -c c23 -t videos  --server EU2 "."
python faceforensics_download_v4.py -d DeepFakeDetection_original -c c23 -t videos  --server EU2 "."


usage: faceforensics_download_v4.py [-h] [-d {original_youtube_videos,original_youtube_videos_info,original,DeepFakeDetection_original,Deepfakes,DeepFakeDetection,Face2Face,FaceShifter,FaceSwap,NeuralTextures,all}]
                                    [-c {raw,c23,c40}] [-t {videos,masks,models}] [-n NUM_VIDEOS] [--server {EU,EU2,CA}]
                                    output_path
