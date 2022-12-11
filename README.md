# Image-Caption-generation

# Full report available here. It was written with other members of the UTS MDSI Deep Learning subject cohort.

https://github.com/Declan-Stockdale/Image-Caption-generation/blob/master/Assignment%203%20Report.docx

## Overview
The following report details the work of our group in developing a deep learning model for captioning images with short descriptions. Each member of the group created a model or series of models, capable of extracting the key features of random images and assigning captions to them. This report will describe the data which was ingested by the models, and then discuss how the data was prepared and used to generate the results. Following this, the various architectures of the different models used will be discussed as well as the different metrics employed to assess the quality of each model for this task. Finally, the best model will be put forward based on the metrics discussed and future recommendations for this project will be presented.

## Dataset
The dataset used to train the models was the Flickr8k image captioning dataset (kaggle.com). This data is split into two main sections, the Flickr8k_dataset, which is a collection of 8092 unique images of random scenes and the Flickr8k_text, which contains various breakdowns of different descriptions for each image. The descriptions are broken down into a training set of 6000 images and their related descriptions, 1000 validation images and another 1000 testing images.

Each image has a collection of 5 descriptions linked to it. While each description is slightly different, they all generally describe the same scene. The following images detail this further:

![image](https://user-images.githubusercontent.com/53500810/206885269-1b36e544-4568-4d2a-abae-0969f97d0646.png)

Since there were nearly 5 descriptions per image, this meant that the corpus was made up of ~40000 different descriptions, each of varying lengths. The following diagram shows the distribution of descriptions, by caption length.

![image](https://user-images.githubusercontent.com/53500810/206885292-022177d3-e27b-4ab3-9262-8f1debc7a945.png)


### Training process
Training of the models was mostly performed on Google Colab using the automatically allocated resources. In addition to this, one of our team members had a substantial GPU capable of running these models relatively quickly. We were then able to bypass Google’s limitations surrounding their resource allocations, as required. All the models used were made available through TensorFlow and its various libraries.

### Data Preparation
In order to train this model, two forms of data preparation were required. Firstly, the image data needed to be configured into a readable format for a convoluted neural network. Secondly, the text descriptions needed to be cleaned and tokenised to create a complete vocabulary of the corpus.
While each member of the team employed different models in their attempts at this project, all of the images were processed through a convolutional Neural Network (CNN). As such, each image needed to be converted into a tensor, with the necessary target size for the specific model being used.

### Process

Some team members ran the initial CNN on the images to extract the relevant features of all the images. These features were then stored in a pickle file to be accessed by the model later, during the compilation process. Fortunately the keras module allows for quick and easy preprocessing, which converts an image into a normalised, numerical multidimensional array, readable by the specific model being employed. As such, the preparation for the image data is very straightforward.
The text data on the other hand requires significantly more set up. First, all associated descriptions need to be clearly mapped to a specific image. With some minor data wrangling, eventually a dictionary with the jpg as the key and a list of all descriptions as the values is created. Next, the text needs to be cleaned and tokenised. The cleaning process is straightforward. First each unique word is placed in a list and then all words are set to lower case, all punctuation is removed, hanging s and a’s are removed, and any numeric characters are removed. Then, before being fed into the Recurrent Neural Network (RNN) model, this list of cleaned unique words is tokenised, resulting in a numbered list of unique words.

Once the list of cleaned and tokenised, the last step was to set up the padding for a sequence so that all token vectors have a uniform length. The max length  caption length for this data set was 34 so all captions less than 34 words long had additional 0’s added at the end to ensure uniform length.

### Architectures tested

A variety of models available from keras were tested for this project. Each model used both a CNN and an RNN. The CNN models were chosen as they performed well on the ImageNet database which comprises a 1000 unique classes. We also preload the ImageNet weights for all our models essentially allowing transfer learning to occur. The CNN serves to extract the features from the images using preloaded ImageNet weights, whereas the RNN generates the sequence of words that make up the caption. The first layer after the input layer in the RNN is an embedding layer. This layer provides a vector space which serves to map the proximity of various words to each other. This is then used to create a viable string of text to caption each image. The following diagram shows the high level architecture of all of our models.

![image](https://user-images.githubusercontent.com/53500810/206885319-205a341e-18ca-4e49-94a4-d1cde851026a.png)


![image](https://user-images.githubusercontent.com/53500810/206885329-743baba9-8b3a-4aea-8970-ec16cf4643cf.png)


### Problems with traditional evaluation metrics
A significant challenge in evaluating automatically generated captions is the sample space of potentially viable solutions. When captioning an image, it is possible to have a variety of different captions which are all technically correct, yet semantically different. Traditional metrics, such as precision, are thus not very useful for assessing how accurate a model is.

This led to the development of alternative metrics to assess model performance. One widely used metric for assessing the accuracy of image captioning is Bi Lingual Evaluation Understanding (BLEU) first published in 2002 (Papineni et al, 2002). This metric is commonly used because it is inexpensive to calculate, language independent and conceptually straightforward. This, along with manually inspecting random samplings of images and their generated captions, is the primary method for evaluation.

BLEU works by separating potential captions into n-grams, that is, ‘n’ length sequences of words, and assessing the accuracy of a model against these different levels. This results in 4 standard metrics, for example, BLEU 1 has n-gram length of 1, BLEU 4 has n-gram length of 4, etc. Each n-gram in the generated caption is compared to the reference caption and the number of matching n-grams is divided by the total number of n-grams in the generated caption. It also has an additional penalty for caption length which improves caption brevity. Importantly, a perfect score is impossible for BLEU. Instead, Google provides the following guidelines


### Results
The following tables contain the BLEU scores of the various models trained. It’s common to report BLEU values from 1-4 in the literature which is also what we’ve reported here.

![image](https://user-images.githubusercontent.com/53500810/206885370-bd7779b9-a33d-4169-8088-eafe61af8c6e.png)

As such, our models have shown that they are capable of generating captions which can occasionally correctly identify and describe the scene depicted in various images. Though there is still significant room for improvement, as is clearly shown by the BLEU-n scores and the mismatched captions in the later images.

We can also observe that the models are learning and improving the quality of the caption with increasing number of epochs. From the image above which used the VGG19 LSTM model, we can see the caption from the model after 1 epoch results in the caption “the child is wearing red hat and holding the face of the water” while the caption fo the model trained for 50 epochs is “girl in pink goggles is swimming in the snow”. It appears that the models are learning information as it was able to identify water but the last model predicted snow which isn’t correct.

### Remaining Issues and Recommendations
This report details the process and the results of training an image captioning model within the time allotted for this task. With more time, there are some avenues that would have been further explored, especially in the hope of increasing the BLEU-4 score on the Flickr8k dataset and creating a model capable of accurately captioning the majority of images.

Firstly, the feature extraction from the CNN models could be further fine tuned to be better suited to this particular dataset. Though a method of validation would be required in order to assess the accuracy. In a similar vein, while the key metric for the evaluation of this project has been BLEU, it may be worthwhile to explore other metrics to give further depth in the assessment of the results.

Another avenue to explore would be the method of word embedding and tokenizing the vocabulary. Currently, the embedding is being handled by keras. However, the results may differ if another form of embedding is used, like for example GloVe or word2vec.

The last immediate path for exploration, is the possible effect of augmenting the incoming image data, as is normally done for image classification to avoid overfitting. In order to potentially better generalise the models, the results of this augmentation could be explored.
As for immediate recommendations for the model in its current state, there is a slight difference in the scores between the models trained with LSTM and GRU. It is recommended that, if ever this model is released to production, a trade off between speed and accuracy needs to be defined. Since GRU is more efficient and quicker, but is generating slightly lower scores, if the model requires speed over accuracy, then the GRU is a good candidate.

Though, to further develop and explore these ideas, the current limitation of computational resources available should ideally be solved. For the development of this report, the majority of the team used Google Colaboratory. While this is a powerful tool with a host of packages already available to use, the limitations on time and GPU usage make training deep learning models very difficult. Fortunately one team member had their own dedicated GPU, capable of training the models at a similar speed to the Colab notebooks. However, ideally all team members should be able to work in an IDE that does log them out after a time limit.

![image](https://user-images.githubusercontent.com/53500810/206885396-46124802-c938-4fa2-bae2-a07657dfa5ff.png)

