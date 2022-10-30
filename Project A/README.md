# Project A


## Introduction 
The goal of this project is to equip you with the tools and technologies required to transfer knowledge from a larger model to a smaller one that can be used practically in real-world settings. Specifically, the project will focus on the setting of Knowledge distillation as a model compression technique. The project
is divided into 2 tasks, (1) using conventional knowledge distillation framework as a model compression method for a popular digit classification dataset, “MNIST” and (2) using transfer learning and knowledge distillation to train a lightweight model for mimicking a pre-trained larger model in a clinical histopathology dataset, “MHIST”. 

## Discussion about knowledge Distillation.

The project is divided into two parts:

1. [Task 1](./Task1.ipynb)
        * What is the purpose of using KD in this paper? 
        KD is used to transfer the knowledge from a whole ensemble of models to a small model that is much easier to deploy.


        * In the paper, what knowledge is transferred from the teacher model to the student model?
        The class probabilities produced by the cumbersome model are used as “soft targets” for training the small model. If the cumbersome model is a large ensemble of simpler models, "soft targets" can be defined as the mean of their individual predictive.


        * What is the temperature hyperparameter T ? Why do we use it when transferring knowledge from one model to another? What effect does the temperature hyperparameter have in KD?
        Temperature is a hyperparameter that is applied to logits to affect the final probabilities from the softmax. 
        In “distillation”, we raise the temperature of the final softmax until the cumbersome model produces a suitably soft set of targets. We then use the same high temperature when training the small model to match these soft targets.
        Using a higher value for T produces a softer probability distribution over classes

        * Explain in detail the loss functions on which the teacher and student models are trained in this paper. How does the task balance parameter affect student learning? 
        The loss function suggested for the teacher model is Cross Entropy loss:
        TeacherLoss = CE(Output of the teacher model, GroundTruth labels) 

        For student loss calculation, KL divergence is the best way to minimize the difference between teacher model output and student model output. So the loss function defined for the student model has an extra term related to the KL divergence:
        StudentLoss = CE(Output of the student model, GroundTruth labels) + Lambda * KL(Output of the teacher model, Output of the student model)

        * Can we look at the KD as a regularization technique, here? Explain your rationale.
        Yes. As mentioned earlier, the class probabilities produced by the cumbersome model are used as “soft targets” for training the small model in KD. Using "soft targets" can be interpreted as label smoothing which is a technique that perturbates the target variable, to make the model less certain of its predictions. Since label smoothing restrains the largest logits fed into the softmax function from becoming much bigger than the rest, it is considered as a regularization technique.
   
2. [Task 2](./Task2.ipynb)

    In this part we responds to questions from the project requirements document [here](./Papers/ECE1512_2022F_ProjectA_Assignment.pdf)

        * Question9:
        This paper suggests an approach for decreasing the gap between the teacher and student models. When this gap gets larger, the efficiency of knowledge distillation (KD) degrades. In other words, less knowledge is transferred from teacher to student model in this situation. To solve this problem, a model named teacher assistant is proposed to fill the gap between teacher and student models. The final framework deploys a multi-step knowledge distillation in which knowledge transfers from teacher to an intermediate-sized network (teacher assistant model), and in the next step, teacher assistant model transfers knowledge to the student. The effectiveness of this idea has been studied in various scenarios. 

        * Question10:
        The larger the gap between teacher and student models means the lower rate of learning. When an an intermediate-sized network is added between these two networks, student learns from TA instead of teacher. Since TAKD scenario bridges the gap between both TA and teacher and TA and student, more knowledge is transfered from teacher to student as a result. 

        * Question11:
        Adding an intermediate-sized network to the conventional KD model will increase the processing time of the network and it will be computationally more expensive. To address this issue, we could choose a small network as the teacher assistant model to reduce the time needs to train it.

# References
