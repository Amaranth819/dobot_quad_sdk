# Deploy RL Controllers on Dobot X1 Hardware


Adapted from walk-these-ways-go2 (https://github.com/Teddy-Liao/walk-these-ways-go2/tree/main)


1. Connect to Dobot X1 robot via ethernet cable, and set the ip address of the laptop to 192.168.5.xxx.
2. Install all requirements.
3. cd deploy/
4. Shut down the sport mode: python kill_robot.py
5. Run ./deploy.sh