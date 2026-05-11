###
 # @Author: York.yangyiny
 # @Date: 2026-03-20 17:13:23
 # @LastEditors: York
 # @LastEditTime: 2026-04-28 17:03:02
 # @FilePath: /E2E/SparseDriveV2/docs/docker_build.sh
 # @Description: Do not edit
 # @Copyright: Copyright (c) 2026 yuyao. All rights reserved.
### 

docker run -it \
  --name SparseDriveV2_yangyy \
  --network host \
  --ipc host \
  --privileged \
  --restart unless-stopped \
  -e QT_X11_NO_MITSHM=1 \
  -e DISPLAY=$HOST_DISPLAY \
   -v /media/backup/yangyy:/home/yangyy \
  -v /etc/localtime:/etc/localtime:ro \
  -v /dev/bus/usb:/dev/bus/usb \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  --device /dev/dri \
  --group-add video \
  --runtime nvidia \
  --dns=8.8.8.8 --dns=8.8.4.4 \
  registry.cn-hangzhou.aliyuncs.com/breton/cuda:11.8.0-cudnn8-devel-ubuntu22.04 \
  /bin/bash