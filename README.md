# comparison-between-Singular-Value-Decomposition-SVD-and-Non-negative-Matrix-Factorization-NMF-
The purpose of this assignment is to compare matrix factorization techniques through a facial recognition study.
For this purpose, the CBCL database, originally used in [3]. Note that
the dataset is divided into train and test datasets. In order to perform your first task, you will use the train dataset. The
techniques from [1] will be used in order to carry out the comparison.

## 1) Implementation and analysis
The following steps correspond to the comparison between Singular Value Decomposition (SVD) and Non-negative
Matrix Factorization (NMF).
# 1. Singular Value Decomposition (SVD)
(a) Use SVD to factorize X as
X = U x V x T ;
where sigma is a 361 x 361 square matrix with diagonal entries delta_i^2, 1 <= i <= 361 = 19 x 19.
(b) Plot the singular values as plot([1:361],diag(sigma)). Then compute the accumulated energy e of the singular
values as: e[i] = sum (delta^2) for 1 <= i <= 361. Afterwards, plot the normalized accumulated energy as
plot([1:361],e=max(e)), where max(e) represents the maximum element of the vector e.
(c) Identify the indices I90, I95 and I99 of e that correspond to the indices when the normalized energy reaches
to 0:9, 0:95 and 0:99, respectively.
(d) Check the corresponding singular faces which are the first I90 columns of U and display them as in Figure
1 of [1]. Comment on the obtained singular faces? Do they have localized or distributed features? Are
they non-negative valued?

![image](https://user-images.githubusercontent.com/36455629/123239990-94ddd800-d4e8-11eb-9021-ecd8d8f9b97e.png)
![image](https://user-images.githubusercontent.com/36455629/123240063-a4f5b780-d4e8-11eb-99d0-7db4b8e35ae0.png)

Defined indices gave the corresponding energy rates as the following;

![image](https://user-images.githubusercontent.com/36455629/123240262-d40c2900-d4e8-11eb-89f6-1294f140142f.png)

Since I_90 = 0, only one singular face is displayed. Singular face does not have any localized but have distributed features, it only has eye, mouth and face shape features. No local feature is obtained from this singular face. There are non-negative values in the plotted singular face.

# 2. Non-negative Matrix Factorization (NMF)

![image](https://user-images.githubusercontent.com/36455629/123241142-91971c00-d4e9-11eb-9b81-4ed0956831d9.png)

![image](https://user-images.githubusercontent.com/36455629/123241215-9f4ca180-d4e9-11eb-843a-02054d50d523.png)

Obtained convergence of iteration plot in Figure 4 is similar to the Figure 3 in the first reference for HALS update label. Since I normalized the input matrix X, obtained plot begins from a lower point than the one in the reference. Normalizing input matrix lessens the processing time and does not change anything in the input matrix.

![image](https://user-images.githubusercontent.com/36455629/123241284-ad9abd80-d4e9-11eb-80d6-5c0a31388a4b.png)

From Figure 5, the first eigenface does not have localized features, it has general nose, eyes, mouth and face shape features. However, second eigenface contains localized features such as front head of the person, third eigenface contains half of the person face so it is localized as well. Last two eigenface contains localized features as cheeks of the person. Therefore, NMF gave localized features not distributed features. Eigenfaces have only non-negative values as described in the first reference.

## 2) Image recovery from noisy data

![image](https://user-images.githubusercontent.com/36455629/123241485-df138900-d4e9-11eb-98bf-eafcf9c4d0f4.png)

![image](https://user-images.githubusercontent.com/36455629/123241543-edfa3b80-d4e9-11eb-8e9b-c40764e75a07.png)

![image](https://user-images.githubusercontent.com/36455629/123241609-f8b4d080-d4e9-11eb-9554-96be672d5dd6.png)

![image](https://user-images.githubusercontent.com/36455629/123241650-ffdbde80-d4e9-11eb-8538-c083a37c6f91.png)

![image](https://user-images.githubusercontent.com/36455629/123241684-066a5600-d4ea-11eb-88ea-4aa9b3a7fbdc.png)

By looking at Figure 6, 7 and 8, it could be said that SVD is a better approach than NMF since blue curves are more likely to converge to 0. Average error is likely to converge to zero as and increases so there is a trend to 0. To lessen the processing time , is normalized and iterations are bounded to 100. Average error would decrease more as the iteration number increases for all noise rates. According to the Figures 6, 7 and 8, noise rate influenced the performance of the two methods. When n=1 and n=10 average error is approximately 10-12 but when 5 error is approximately 15-20. However, for different noise rates SVD always gave better results than NMF.

## 3) Image recovery from masked data

![image](https://user-images.githubusercontent.com/36455629/123241910-36195e00-d4ea-11eb-93bb-786b7635d7d2.png)

![image](https://user-images.githubusercontent.com/36455629/123241957-3f0a2f80-d4ea-11eb-9581-589567c17538.png)

By looking at Figure 9, it could be said that SVD is slightly better approach than NMF since blue curves are more likely to converge to 0. In this task, SVD and NMF approaches gave more similar results than in the second part’s task. The average error difference was more obvious in Figures 6, 7 and 8 but in Figure 9 the difference is not that obvious.
Average error is likely to converge to zero as r_nmf and r_svd increases so there is a trend to 0. Average error would decrease more as the iteration number increases. However, for different noise rates SVD always gave better results than NMF.
