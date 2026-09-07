export LD_LIBRARY_PATH="/usr/local/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CYCLONEDDS_URI="file://$PWD/cyclonedds.xml"

# echo "Shutting down the sport mode"
# python kill_robot.py 192.168.5.2:50051

echo "Start deployment"
python -u deploy.py