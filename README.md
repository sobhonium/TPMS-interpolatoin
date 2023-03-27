# TPMS-interpolatoin


In this rep, Deepsdf (with an AutoDecoder) is trained to give an interpolation representation for Minimal surfaces. To see the NN architecture see [this](./model/) file.
The attempt is to have an interpolation between minimal surfaces like Gyroid,Primitive, FisherS. 

There are some limitations in terms of this interpolation.
- The volumes should be continous,
- The volumes should be symetrical in all axes.


![image](/results/results-plot.png)
