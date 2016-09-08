
"""
Molecular Dynamic (M.D) with tensor flow: This code demonstrate how to build molecular dynamic simulation using tensorflow. 
TensorFlow is an open source software library for numerical computation using data flow graph,
It mostly used for machine learning but can be used to run any system as that can be written as dataflow graph. The code involve building data flow graph for single simulation step of molecular dynamics.
This basically mean building the molecular dynamic simulation using tensor operations only. This this simulation step graph in runned in a loop to create multistep molecular dynamic model. The particles location is plotted on screen in each step.

The data flow graph of the M.D simulation involve the following steps:
1) Calculate distance vector between every pair of particles in the system.
2) Calculate force between every pair of particles in the system.
3) Calculate acceleration for each particle in the system.
4) Update particle's speed using accelerating.
5) Update particles position using their new speed.
6) Reduce speed of all particles by some factor to induce cooling effect.

Input Parameters:
InitPosition: Initial position of all particles in format [[x1,y1],[x2,y2],[x3,y3]...].
InitVelocities: Initial velocities of all particles in format [[vx1,vy1],[vx2,vy2],[vx3,vy3]...].
dt: Time step for the simulation if this too big the simulation will explode.
m:Particle mass.
CoolingRate: Increase/decrease the speed of all particles by factor to achieve cooling/heating effect.
PeriodicBoundary: Tell the weather the system use periodic boundary.
CellSize: If the system use periodic boundary condition what the cell size.

Tensorflow graph parameters:
x,v: Placeholders that contain the input locations and velocities of of particles.
xnew,vnew: Output Particle's location and position in the end of the simulation step. 
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
#------------------------------INPUT PARAMETERS-----------------------------------------------------------------
PeriodicBoundary=True# does the simulation use preiodic boundary if so make sure to set cell size paramter
CellSize=1.2 # Size of cell in simulation important if you use priodic boundary conditions
dt=tf.constant(0.00005,tf.float32)# Time step: time lapse of molecular dynamic simulation step (if this too big the simulation will explode if its too small it can take lots of time)
m=tf.constant(0.5,tf.float32)# mass of particles
NumParticles=100 # Number of particles to generate
InitPosition=np.random.rand(NumParticles,2)*CellSize## initial position of particles
InitVelocities =np.random.rand(NumParticles,2)*10 # initial velocities of particles
CoolingRate=0.995# Increase/decrease the speed of all particles by factor to achive cooling/heating effect.
#---------------------Placeholders input parametes for the graph---------------------------------------------
x = tf.placeholder(dtype=tf.float32)# Position of particles in the begining of simulation step (input)
v = tf.placeholder(dtype=tf.float32)# Velocities of particles  in the begining of simulation step (input)
CoolingFactor = tf.placeholder(dtype=tf.float32)# The speed of each particle will be increase/decrease by this number to simulate cooling/heating effect
#***************Create the graph to simulate single molecular dynamic (m.d) simulation step---------------------------------------------------------------------------------------------------
expanded_x1 = tf.expand_dims(x, 0)# The new dimesnion is of undefine size and hypotetically each element in this dimennsion is the same as the original element  the size of the new dimension will be determined once this is used
#----------------------------------Create priodic boundary by generating replicas of the cell from every side--------------------------------------------------------------------
if PeriodicBoundary: # If you use priodice boundaries this step calulate the location of the particles in the neighbours cell and add them to the force calculation this assume 2d system if you use 3d another for loop need to be added
   CellPos=[]#Generate cordinates particles of all the neighbors cells (by tranlating the particles coordinates)
   for i in range(-1,2):#Generate cordinates of all the neighbors cells
    for i2 in range(-1,2):
        CellPos.append([i*CellSize,i2*CellSize])#Generate the corner (0,0) position of each of the neighbouring cells
   AllParticles= tf.expand_dims(CellPos, 1)+expanded_x1  #Generate the location of the particles in neighbouring cells by adding particles location in center cell with the location of all the neighbors cellls 
   AllParticles=tf.reshape(AllParticles,[-1,2]) # Reshape to single array of particles coordinates
   expanded_x2 = tf.expand_dims(AllParticles, 1)# The new dimesnion is of undefine size and hypotetically each element in this dimennsion is the same as the original element  the size of the new dimension will be determined once this is used
#-----------------------For none priodic boundaries conditions----------------------------------------
else:# incase none priodic boundary conditions are used.
   expanded_x2 = tf.expand_dims(x, 1)# The new dimesnion is of undefine size and hypotetically each element in this dimennsion is the same as the original element  the size of the new dimension will be determined once this is used
#-----------------------------------------------------------------------------------
rx=tf.sub(expanded_x1,expanded_x2 )#Distance between every pair of particles in x in every dimension (dx,dy)
rx2=tf.square(rx) # sqar distane for each particle pair in each dimension  (dx^2,dx^2)
r2=tf.reduce_sum(rx2,2) # absolute squar distance between every pair of particles(dx^2+dx^2)
r=tf.sqrt(r2) # absolute distance between every pair of particles

r=tf.maximum(r,tf.ones_like(r)*0.02)# To avoid division by zero make min distance larger then 0 this add to prevent simulation explosion if particles get too closed
F=-30/tf.pow(r,2)+10/tf.pow(r,3) # Force between pair of particles F=9/r^2-1/r^3 (attracion 9/r^2 and repulsion 1/r^3)
Fx=tf.mul(rx,tf.expand_dims(F/r,2))# The forces per axis applied between each pair of particles we divide the force by r since rx is not normalize by distance between particles
Accel=tf.scalar_mul(dt/m,Fx)# Acceleration resulted from forces between each pair is simply force between each pair divide by particle map and multiply by time of step
dv=tf.reduce_sum(Accel,0) ## or dim2? Sum velocity  changes for each particle in each step of the simulation
vnew=(v+dv)*CoolingFactor# Update velocity for particle U=
xnew=x+(vnew)*dt# Update position for each particles according to particle speed (avereged on new and previous speed)
#-----------if epetitive  boundary conditions are used make sure particle poistion dont exceed cell size-------------------------------------------------------------------
if PeriodicBoundary:
   xnew=tf.mod(xnew+CellSize,CellSize)# repititive boundary conditions make sure the particle never exit the box
#---------------------------initiate Drawing---------------------------------------------------------
plt.ion() # make sure plot can be update
fig=plt.figure()# start plot
#---------------------------Run the graph---------------------------------------------------------------------------
with tf.Session() as session: #Create graph session
     for i in range(1000):   
         #Run single graph  (sigle simulation step)       
         [InitPosition,InitVelocities]=session.run([xnew,vnew],feed_dict={x: InitPosition,v:InitVelocities,CoolingFactor:CoolingRate})#Run Graph calculate new velocities and speeds
         #----------------------Plot particles position real time---------------------------------------------       
         if (i%1==0): 
           Cord=np.array(InitPosition) # Tranfer particles cordinates to numpy array format to use in plotting functin
           plt.clf()# clear figure   
           plt.xlim(0, CellSize)# define figure axis size
           plt.ylim(0, CellSize)# define figure axis size
           plt.title(["step ",i])# figure  title
           plt.scatter(Cord[:,0],Cord[:,1]);# add particle position to graph
           plt.show()# show on screen
           plt.pause(0.02) #time delay
