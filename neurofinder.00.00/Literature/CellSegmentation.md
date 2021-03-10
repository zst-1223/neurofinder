# 1. Cell Detection

## 2016[Cell]Nucleus localized
<u>2016[cell]The Serotonergic System Tracks the Outcomes of Actions to Mediate Short-Term Motor Learning</u>
==Takashi & Misha==

> github: https://github.com/ahrens-lab/Kawashima_et_al_Cell_2016/

**H2B-GCaMP6f**: calcium indicator ---> (predominantly) nucleus-localized

![image-20200901150607564](assets/image-20200901150607564.png)

(from figure S2 B)

1. Original  Image
2. **GCaMP-positive areas** of the volume were coarse-extracted by **binary thresholding** based on <u>absolute pixel intensity and local image contrast</u>.
3. **Local intensity normalization** [radius of 3.2um]: a relative rank of pixel intensity (0(d)-1(b))
4. **average kernel** [r: 1.2um]: smoothing
5. **maximum value**[r: 2.4um]: centroid
6. **time series of pixel intensity**[r: 1.2um]

<font color=red size=5> Here we repeat step4 and step5 twice to improve accuracy </font>

- Approximately $79\% \pm 7\%$ of neurons identifiable by eye were automatically recognized; 
- Approximately $89\% \pm 5\%$ of recognized objects matched with eye-identifiable neurons;

### Possible direction

1. Average image VS Random sampling images
	- one image is restricted
	- several images to denoise the results
2. Model-based methods for automatically cell detection 
	- Here we have labeled data here? (if we can obtain)
3. <font color=red> Detect the trajectory of nucleus directly</font>

### Parameters

- contrast threshold=7: threshold for local contrast. This sets how the cell should be brighter than surrounding darkest point.
- binary threshold=170: minimum brightness threshold for the cell to be detected
- cell radium=4: radius of cells to be detected

### Questions

1. Whether GCaMP has influence on the data analysis?

## 2014[NeuralNet]NMF

<u>2014[NeuralNet]Detecting cells using non-negative matrix factorization on calcium imaging data</u>

==Ryuichi Maruyama & Toru Aonish==

### Abstract

- bleaching line of the background fluorescence intensity
- NMF performs well than PCA
- rapid transient components corresponding to somas and dendrites of many neurons

$$
F=SA+s_ba_b+\mathrm{noise}
$$

### Algorithm

![image-20201010162330272](assets/image-20201010162330272.png)

### Pipeline

#### Preprocessing

- low-pass filtering
- background fluorescence (manually locate a large ROI at the background region)
- Application of NMF method

#### ROI selection

- thresholding

#### Merging segmented sections

- Merge ROIs with similar fluorescence

![image-20201103114558229](assets/image-20201103114558229.png)

### Limit

- estimation of background
- localization constrains(Boolean or value with threshold or cell size)

## 2019[Cell]NMF

<u>2019[cell]Glia Accumulate Evidence that Actions Are Futile and Suppress Unsuccessful Behavior</u>

==Yu Mu & Misha==

> github: https://github.com/mikarubi/voluseg

### Methods

Objective: population of cell bodies and neuropil(神经纤维)

1. Preprocessing data for NMF
	- Intensity-based brain mask
	- 1000 spatially contiguous three-dimensional blocks (overlapped)
2. Constrained non-negative matrix factorization
	For $n$ voxels, $t$ time points, and $c$ cells, we factorized,
$$
V^{n\times t} \approx W^{n\times c} H ^{c\times t} + X^{n\times 1} I ^{1\times t}
$$
where:
- V: spatiotemporal fluorescence matrix for each block
- W: spatial location 
- H: time series of segmented cells
- X and I: rank-1 spatiotemporal model of the background signal

3. Cell numbers

- Assume each block would contain 100 tightly packed 6um diameter spheres
- Set number of cells in each block to be 100
- gradually reduce the number by iterative multiplying 0.95 until converging

### <font color=red> Initialization</font>

- Local intensity peaks and local correlation coefficients to initialize $W$ (clustering)
- Constant value to initialize $X$ 
- I: average of mask value
- Alternating least squares with at least 10 iterations and at most 100 iterations or until numerical convergence (residual < 0.001)

### <font color=red>Spatial and temporal constraints</font>

- mean-amplitude normalization of each cell
- hard spatial constraints (< 12um) centered on a local-intensity peak (mask)
- soft sparseness constraints: spatial footprint for each cell (> 6 um)
$$
sparseness(W_i) = \frac{\sqrt{n} - \sum_j |W_{ij}|/\sqrt{\sum_j|W_i|^2}}{\sqrt{n}-1}
$$

### Code

> step0: process parameters and create parameter file
>
> step1: process original images and save into nifti format
>
> step2: register images to a single middle image
>
> step3: create intensity mask from the average registered image
>
> **step4: detect cells in images**
>
> **step4a: get coordinates of individual blocks**
>
> **step4b: load timeseries in individual blocks, slice-time correct, and find similar timeseries**
>
> **step4c: initialize cell positions in individual blocks**
>
> **step4d: cell detection via nonnegative matrix factorization with sparseness projection**
>
> **step4e: collect cells across all blocks**
>
> **step5: remove noise cells, detrend and detect baseline**



## 2015[NIPS]SSTD

<u>2015[NIPS]Sparse space-time deconvolution for Calcium image analysis</u>

==Ferran Diego & Fred A. Hamprecht (HCI)==

### Methods

non-negative matrix factorization

- cell segmentation and activity estimates without the need for heuristic pre- or postprocessing
- locations of cells 
- their activity along with
- typical cell shapes
- typical impulse responses.

![image-20200910182417913](assets/image-20200910182417913.png)

## 2016[Neuron]Simultaneous Denoising, Deconvolution, and Demixing of Calcium Imaging Data

==Eftychios A. Pnevmatikakis==

### Abstract

- simultaneously **identify the locations of the neurons**, **demix spatially overlapping components**, and **denoise and deconvolve the spiking activity** from the slow dynamics of the calcium indicator
- CNMF

### Constrained Sparse Nonnegative Calcium Deconvolution

> Deconvolving the neural activity from a time series trace of recorded fluorescence

#### Autoregressive Model

$$
c(t)=\sum_{j=1}^{p}\gamma_jc(t-j) + s(t)
$$

- $c(t)$: calcium concentration dynamics
- $s(t)$: the number of spikes at t-steps

#### Observed fluorescence

$$
y(t)=\alpha(c(t)+b)+\epsilon(t), \ \epsilon(t)\sim\mathcal{N}(0, \delta^2)
$$

- $\alpha$: non-negative scalar
- b: baseline concentration
- $\epsilon$: noise 

## CaImAn [CNMF]

<u>2019[eLife]CaImAn an open source tool for scalable calcium imaging data analysis</u>

==Eftychios A. Pnevmatikakis==

### constrained sparse nonnegative calcium deconvolution

- approximate the **calcium transient** as the impulse response of an autoregressive (AR) process of order p //// estimated by adapting standard AR estimation methods
- estimate the spiking signal by solving a non-negative, sparse constrained deconvolution (CD) problem ///// convex

### Cell segmentation



### Reference

[1] https://github.com/flatironinstitute/CaImAn-MATLAB
[2] https://github.com/flatironinstitute/CaImAn (Python pkg)

## 2019[PNAS]Spatiotemporal deep learning

<u>2019[PNAS]Fast and robust active neuron segmentation in two photon calcium imaging using spatiotemporal deep learning</u>

> https://github.com/soltanianzadeh/STNeuroNet

==Somayyeh Soltanian-Zadeha & Sina Farsiua==

### Abstract

- 3D CNN (DenseVnet)
- high accuracy: 80% in ABO
- faster: training 11.5h / prediction:171.4s

### Reference

1. Neurofinder challenge [http://neurofinder.codeneuro.org/][https://github.com/codeneuro/neurofinder]
2. Allen Brain Observatiry (ABO) [http://observatory.brain-map.org/visualcoding]

# 2. Basic approach

## 2001[NIPS]Algorithms for Non-negative Matrix factorization

==Daniel D Lee & H. Sebastian Seung (MIT)==

### Abstract

- minimize the conventional least squares error while the other minimizes the generalized KL divergence
- auxiliary function analogous
- Expectation Maximization algorithm (proof of convergence)

### Methods

#### Non-negative matrix factorization (NMF)

> Given a non-negative matrix $V$, find non-negative matrix factors $W$ and $H$ such that

$$
V \approx WH  \tag{2-1}
$$

#### Loss function

**Euclidean distance**

$$
||A-B||^2=\sum_{ij}(A_{ij} - B_{ij})^2 \tag{2-2}
$$

**KL divergence**

$$
D(A||B) = \sum_{ij}(A_{ij}\log\frac{A_{ij}}{B_{ij}}-A_{ij}+B_{ij}) \tag{2-3}
$$

#### Multiplicative update rules

> **Theorem 1** The Euclidean distance $||V-WH||$ is nonincreasing under the update rules
> $$
> H_{\alpha\mu}\leftarrow H_{\alpha\mu}\frac{(W^TV)_{\alpha\mu}}{(W^TWH)_{\alpha\mu}} \\
> W_{i\alpha}\leftarrow W_{i\alpha}\frac{(VH^T)_{i\alpha}}{(WHH^T)_{i\alpha}} \\
> $$
> 

> **Theorem 1** The Euclidean distance $D(|V||WH)$ is nonincreasing under the update rules
> $$
> H_{\alpha\mu}\leftarrow H_{\alpha\mu}\frac{\sum_iW_{i\alpha}V_{i\mu}/(WH)_{i\mu}}{\sum_kW_{k\alpha}} \\
> W_{i\alpha}\leftarrow W_{i\alpha}\frac{\sum_{\mu}H_{\alpha\mu}V_{i\mu}/(WH)_{i\mu}}{\sum_vH_{\alpha v}} \\
> $$
> 

#### Proofs of convergence

> If G is an auxiliary function ($G(h,h')\ge F(h), G(h,h)=F(h)$),then F is non increasing under the update
> $$
> h^{t+1}=\arg\min_hG(h,h^t)
> $$

![image-20201010152044146](assets/image-20201010152044146.png)

## 2004[JMLR]NMF with sparseness constraints

==Patrik O.Hoyer==

### Definition of Sparseness

$$
\mathrm{sparseness}(x)=\frac{\sqrt{n}-(\sum|x_i|)/\sqrt{\sum x_i^2}}{\sqrt{n}-1}
$$

This function evaluates to unity if and only if x contains only a
single non-zero component, and takes a value of zero if and only if all components are equal (up to signs).

![image-20201103164517967](assets/image-20201103164517967.png)

### Algorithm

![image-20201103164712567](assets/image-20201103164712567.png)

### Projection operator

![image-20201103164805452](assets/image-20201103164805452.png)

### Result

![image-20201103165038968](assets/image-20201103165038968.png)

## 2017[IJCAI]Deep Matrix Factorization

==Hong-Jian Xue && Jiajun Chen==

