# Rocket_Game_Neural_Network
It is a game that player flies a rocket to successfully land on a platform without crashing or going out of the frame.

How to start the game:

1) Open up a terminal in side of the Rocket_Game_Neural_Network folder.
2) Run python Main.py
3) Game start with a Menu with 3 options
   1) Play Game:
      Player just plays the game without any data collection. It is just for fun.
   2) Data Collection:
      Player plays the game to collect data. Data is collected under ce889_dataCollection.csv file. This repository has already had data file inside of ce889_dataCollection.csv. So there is no need to collect       data. If user wants to collect a data from scratch just back this data up in different file and empty ce889_dataCollection.csv file and start playing the game on Collect Data mode. Data will include 4         columns y-axis location, x-axis location, speed on y-axis, speed on x-axis respectively. Game will collect the data as long as rocket flies each turn. Longer the rocket stays up more data will be              collected. However, this will affect the data quality negatively.
   3) Neural Network:
      User test the Neural Network quality on this mode. The rocket will be played by AI.
      Train.py includes Neural Network training. After this file is run, model turns a pair of array that is output of weights of the Neural Network.
      User should put these arrays that are taken from Train.py inside of the NeuralNetHolder.py to run Neural Network AI properly.
