# Homework 3

Tyler Amos

26 May 2019



# Q3

For code portions (a-c), please see the attached files. Note that I modified the code so it takes an additional option, whether to use period resampling based on a fixed number of iterations or a number of effective particles. 


# Q3 d - f

I noticed in order to have a reasonable approximation of the ground truth, it was necessary to reserve some particles to be resampled according to a uniform distribution, not the likelihood-informed distribution. This helped to avoid particle depletion. In terms of number of particles, I found a relatively small number of particles was best as it allowed the filter to run quickly. Increasing the number of particles over 1,000 seemed to greatly slow down computation. I found a number of particles in the neighbourhood of 600-800 was usually sufficient to have a reasonable approximation. When resampling every time, it was necessary to have the most particles (~800), while resampling intermittently required less (~650), and resampling when the number of effective particles dropped below a threshold required the least (~550 - 700)

I also noticed the filter seemed able to identify the 'type' of location it was in more than the exact place. For example, the filter might indicate the robot's position as the southwest corner when the robot was in fact in the southeast corner. This would suggest the filter is identifying the presence of a corner, and even some elements of the robot's orientation. The filter often improved its estimate when the robot took circuitous paths. I would hypothesize this is because the change in bearing and linear movement allow for more precise estimates of the robot's position. 

I found not resampling at each step was helpful for both avoiding particle depletion and improving the estimate of the robot's pose. I hypothesize this is because the sucessive prediction update steps encodes more information about the movement model's likelihood that simply resamping each time. Even better than resampling at a fixed interval was using the number of effective particles to trigger resampling. In general the estimate of the robot's pose was much better when using the number of effective particles and computation was much quicker. 

With respect to the 'kidnapped' robot problem specifically, I found a similar number of particles (200-400) provided a measure of convergence (my random un-weighted resampling ensures there is not complete convergence). A higher value (e.g., 1,000) would usually provide better performance, but would be substantially slower than these smaller numbers. 

