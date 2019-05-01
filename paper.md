# Paper 1 CityFlow: A City-Scale Benchmark for Multi-Target
Multi-Camera Vehicle Tracking and Re-Identification
## key
- 1) provide annotations for the original videos,the camera geometry and calibration information
- 2) provide spatio-temporal information can be leveraged to resolve ambiguity in image-based REID
- 3) open the way for new research problems such as vehicle pose estimation,viewpoint generation
## 3 CityFlow benchmark
### 3.3 Subset for image-based ReID
## 4 Evaluated baselines
### 4.3 Spatial-temporal analysis
- 1) Liu et al. [30] propose a progressive and multimodal vehicle ReID framework (PROVID)
- 2) a method based on two-way Gaussian mixture model features (2WGMMF) [21] achieves state-of-the-art accuracy on the NLPR MCT benchmark [8] by learning the transition time between camera views using Gaussian distributions
- 3) In FVS [45], however, since no training data is provided, the temporal distribution is pre-defined based on the estimated distance between cameras
# Paper2 Vehicle Re-Identification: an Efficient Baseline Using Triplet Embedding
## KEY
- Batch sample
## QA
- Market-1501 dataset provides an additional 500k distractors recorded at another time.
## NOTES
- an output-normalization layer actually hides problems in the training, such as slowly collapsing or exploding embeddings. We did not use a normalizing layer in any of our final experiments
## 5 Results and Discussions
### 5.1 VeRi
- Adding a normalized layer performs poorly for the triplet loss. This is also reported by [10] wherein normalized layer could result in collapsed embeddings
#### Comparsion to the state of the art approaches
- GSTE,R[1],achieve better top-k acc
- VAMI+ST,R[52]
- Path-LSTM,R[37]
# Paper 3 In Defense of the Triplet Loss for Person Re-Identification
## 3.2 Training
### Comparison to state-of-the-art
- DTL,R[8]
# Paper 4 Viewpoint-aware Attentive Multi-view Inference for Vehicle Re-identification
- Viewpoint-aware Attentive Multi-view Inference Model(VAMI)
## 1 Introduction
- make use of license plate
- spatial-temporal information[15,26,23]
- a general model only based on visual appearances
- Three main contributions
    - a viewpoint-aware attention model is proposed to obtain attention maps from the input image
    - Given the attentive features of a single-view input,we design a conditional multi-view generative network to infer a global feature containing different viewpoints’ information of the input vehicle.
    - In addition to inferring multi-view features, we embed pairwise distance metric learning in the network to place the same vehicle together and push different vehicles away.
## 2 Related Work
- Attention Mechanisms
- Generative Adversarial Network
- Propose an attentive multi-view feature generative network by adversarial learning
## 3 Proposed Methods
### TODO
- 3.2.3 Adversarial Multi-view Feature Learning
# Paper 5 Deep Relative Distance Learning: Tell the Difference Between Similar Vehicles
## Abstract
We propose a Deep Relative Distance Learning (DRDL) method which exploits a two-branch deep convolutional network to project raw vehicle images into an Euclidean space where distance
can be directly used to measure the similarity of arbitrary two vehicles.
## 2 Related Work
- Yi et al.[22] applied a “siamese” deep network which has a symmetric structure with two sub-network to learn pair-wise similarity.
## 3 Approaches
- In stage 1, instead of just considering the simple pairwise relations between two queries, our proposed approach first generates a candidate visual-spatio-temporal path with the two queries as starting and ending states
- In stage 2, the candidate visual-spatiotemporal path acts as regularization priors and a SiameseCNN+path-LSTM network is utilized to determine whether the queries have the same identity
# Paper 6 Orientation Invariant Feature Embedding and Spatial Temporal Regularization for Vehicle Re-identification
## 0 Abstract
With orientation invariant feature embedding,local region features of different orientations can be extracted based on 20 key point locations and can be well aligned and combined
- spatial-temporal regularization
- both the orientation invariant feature embedding and the spatial-temporal regularization achieve considerable imporvements
## 1 Introduction
## 2 Related Work
- Fine-grained vehicle model classification
- Object key point localization
    - Our proposed method shows that vehicle key points can guide the learning and alignment of local regions in input vehicle images and improve the overall vehicle ReID performance.
## 3 Methodology
### orientation invariant feature embedding component
### spatial temporal regularization component
### Regularization by spatio-temporal Modeling
- appearance features may not be adequate enough to distinguish one vehicle from others
## 4 Experiments
### 4.2 Evaluation results
- Bag of Words with Color Name Descriptor(BOW-CN)
# [TODO]Paper 7 LARGE-SCALE VEHICLE RE-IDENTIFICATION IN URBAN SURVEILLANCE VIDEOS
# Paper 8 PROVID: Progressive and Multi-modal Vehicle Re-identification for Large-scale Urban Surveillance
## 1 Introduction
- detection
- fine-grained categorization
- 3-D pose estimation
- driveer behavior modeling
# Paper 9 Viewpoint-aware Attentive Multi-view Inference for Vehicle Re-identification


