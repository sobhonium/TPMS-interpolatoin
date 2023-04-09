# TPMS-interpolatoin


In this rep, Deepsdf (with an AutoDecoder) is trained to give an interpolation representation for Minimal surfaces. To see the NN architecture see [this](./model/) file.
The attempt is to have an interpolation between minimal surfaces like Gyroid,Primitive, FisherS. 
![image](shapes.png)


The trained model finds the results like bellow:

![image](/results/results-plot.png)

Code (1,0,0), (0,1,0) and (0,0,1) are Gyroid, primitive and Fisher respectively.  
There are some limitations in terms of this interpolation.
- The volumes should be continous (see image bellow),
- The volumes should be symetrical in all axes (see image bellow).

![image](/results/results-plot2.png)



## Acknowledgments

We would like to thank ...

## Coding style

*   tab for indentation
*   80 character line length
*   PEP8 formatting

## Disclaimer

This is not an officially supported ....
