

# EcoFed: Efficient Communication for DNN Partitioning-based Federated Learning

### About the research
Efficiently running federated learning (FL) on resource-constrained devices is challenging since they are required to train computationally intensive deep neural networks (DNN) independently. DNN partitioning-based FL (DPFL) has been proposed as one mechanism to accelerate training where the layers of a DNN (or computation) are offloaded from the device to the server. However, this creates significant communication overheads since the intermediate activation and gradient need to be transferred between the device and the server during training.
While current research reduces the communication introduced by DNN partitioning using local loss-based methods, we demonstrate that these methods are ineffective in improving the overall efficiency (communication overhead and training speed) of a DPFL system. This is because they suffer from accuracy degradation and ignore the communication costs incurred when transferring the activation from the device to the server. This article proposes EcoFed - a communication efficient framework for DPFL systems. 

EcoFed eliminates the transmission of the gradient by developing pre-trained initialization of the DNN model on the device for the first time. This reduces the accuracy degradation seen in local loss-based methods.
In addition, EcoFed proposes a novel replay buffer mechanism and implements a quantization-based compression technique to reduce the transmission of the activation. 
It is experimentally demonstrated that EcoFed can reduce the communication cost by up to 133x and accelerate training by up to 21x when compared to classic FL. Compared to vanilla DPFL, EcoFed achieves a 16x communication reduction and 2.86x training time speed-up.

### Code Structure
The repository contains the source code of EcoFed.

TBD

### Setting up the environment
TBD

### Citation
TBD
