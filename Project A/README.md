# Project A

1.  Read the paper “Distilling the Knowledge in a Neural Network” [6] carefully, and then answer the following questions: [2.5 Marks]
    * What is the purpose of using KD in this paper? 
    KD is used to transfer the knowledge from a whole ensemble of models to a small model that is much easier to deploy.


    * In the paper, what knowledge is transferred from the teacher model to the student model?
    The class probabilities produced by the cumbersome model are used as “soft targets” for training the small model. If the cumbersome model is a large ensemble of simpler models, "soft targets" can be defined as the mean of their individual predictive.


    * What is the temperature hyperparameter T ? Why do we use it when transferring knowledgefrom one model to another? What effect does the temperature hyperparameter have in KD?
    Temperature is a hyperparameter which is applied to logits to affect the final probabilities from the softmax. 
    In “distillation”, we raise the temperature of the final softmax until the cumbersome model produces a suitably soft set of targets. We then use the same high temperature when training the small model to match these soft targets.
    Using a higher value for T produces a softer probability distribution over classes

    * Explain in detail the loss functions on which the teacher and student model are trained in this paper. How does the task balance parameter affect student learning? 

    * Can we look at the KD as a regularization technique, here? Explain your rationale.
    Yes. As menthiend earlier, the class probabilities produced by the cumbersome model are used as “soft targets” for training the small model in KD. Using "soft targets" can be interpreted as label smoothing which is a technique that perturbates the target variable, to make the model less certain of its predictions. Since label smoothing restrains the largest logits fed into the softmax function from becoming much bigger than the rest, it is considered as a regularization technique.
   