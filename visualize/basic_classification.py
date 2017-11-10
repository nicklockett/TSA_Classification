from classes import *
from dataExtraction import *
from random import shuffle
from sklearn import svm, metrics, tree, naive_bayes
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AffinityPropagation

from basic_cnn import extract2DDataSet

import matplotlib.pyplot as plt

block_size = 44
# Set segmentNumber
segmentNumber = 0.8

sc = SupervisedClassifier('../../stage1_labels.csv')

# need to ingest and iterate through multiple bodies
image_path_list = ["../../../rec/data/PSRC/Data/stage1/a3d/831600b1b6984119fc87529bf4b61ade.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/4b28d86db0cedb95323986f74e8fc894.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/b93f543e66317a22645be549dc1c008a.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/e253adde4a0a6733cbf12a8e8a17d2d0.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/3a8696b99b2d1b28be62389d48d697be.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/7a465515247d5150a437499ed4dd31a8.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/b2aef1eb2d498341049cd1ca6c99e06e.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/137cd2942a4d022921fda492ff79d40f.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/b60c5e2b4e6faafccbdc2e6893428481.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/0ada538288f0b62b01510b397a8acb9e.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/fbd3ca2d2af1785bfe3af2629f35bc3f.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/f6e4d412642e5cc4fcb0f6a08b592a04.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/fb07ddc68370ee1a59ce61a50baf6ae3.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/8d5861df3b346dbc595694a39152b4fd.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/7142e2ff6b927d5154afded4c90e2acd.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/ae783a8040b1c8c0e729e787097e4113.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/e78e46f1dc5490ffa8925bc6c277b560.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/832ea189a29a25ec10731796599231ba.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/ee58b1e752416fbd0c0928d1a49a13a6.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/3e5e4ede1b6a4ec767480b261336b1b5.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/b220ad927a2a32a1e5bd329c4960a443.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/74244553005b05ece2981f71bcab30f7.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/7d8935f1992f7df23cb5d1cf2c987a63.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/dc834b5d53aaab3b7cc909a101f3c263.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/8a3a0461a76394d6d2ae82cc34cb11c5.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/8464ffead51e673a5244a1b5c0c63b86.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/fd23dddfcd06dcb4dacc1731a348b8cd.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/3c0668db35915783be0af87b9fa53317.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/5e429137784baf5646211dcc8c16ca51.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/e330c71147b29b83782fb79b219efa0f.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/84312f6fe146f36c0da50a60cb4b3b70.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/4881eeb549b78b3fe90f9cdde9a5f649.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/b02f6791d2b027a042de141562fc345a.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/690e67cf1122bbe0a851f23498a38c39.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/df788b51010c6de82fd1a3ff755ddf31.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/88de0795410e7fe94d1c86868a3ca053.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/172acab0b917b44f946c94051feab878.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/499aa07eb418b3d46209d5f937d45d67.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/32d505addd921c1f6fefb7ddc51e7f94.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/86d97d6bb6bbfd6542954ebbdb2ff28e.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/011516ab0eca7cad7f5257672ddde70e.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/8085d6bf5256a67fd53b545dbb53caf4.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/e299540987bf5ccf60ec2f2562ff67b1.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/b5a09d3568108559bae352404708e1c9.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/f6f0834b756940c73a4424cedeaee3b1.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/6307c2e0b60d78fd104b90b0f26ad0df.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/71a28dc56c2039c74eba50a79831bfee.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/e9e2bc4f4f319943935551d502fc11c2.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/43d1a815858d7584eb6e0e84f8a1bf8b.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/48f1d872def62e5d5150c1a7d64d1196.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/e06b9551603d329dfe720f67e37ce87b.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/eb3e1410bfa1f034f6bed038684cfc29.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/5cb8d7ef7176edcd0dd38e4c9921f185.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/690c0567ddd681f71652841feab25cea.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/ec9c7903d4665303f7d3150399af8d84.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/84e2375926811ef6ca836e2bd172997f.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/471829836c7df1fa0c63721d09ea6db9.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/6c4acb4d8cc81e0ce4f18a3cb66a9791.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/e2b431b503c61a1044815b9c53f3fe4f.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/da92d0aae2f5450422d6306d450b90a0.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/e38dcb351fe15acd74d1b0352c0e0e48.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/8db2645372fe8c4fcb62319a70deb914.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/e195068859c5847b028c905dd1ccb81d.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/b53eee6d51543953cdc99756df81e3fb.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/0fdad88d401b09d417ffbc490640d9e2.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/1f0ecd9585289e8848549575223caaeb.a3d","""
"../../../rec/data/PSRC/Data/stage1/a3d/d78bcc549aaa09af770f2a05fbc97e5b.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/401dc40c30b9206019d61c4ed625cfce.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/416a6888eb7ad8c4416fe0b620435136.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/fb397be3a8a1e9312d8b97a95bc97ae6.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/818281be1ffe4daa57fe8fc31b05b0d7.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/ea729b325bbf9762d2d31ba6dbae62b3.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/01941f33fd090ae5df8c95992c027862.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/f2c1d30f352f6b5ab8dd5da31f85ee1d.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/13bc14ee7bc8d31a150b0744a6f1e0fe.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/b41ded8fb8ebac447cbe0c50574fc9ed.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/48309fc9d12a28f0ba94b8c6d81bc3c0.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/e356161eeeadb1bbdeda6644e244e8fc.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/adf2e22d579cbfb345585ca757c4db58.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/f58f9f63785b5e7c996cacab2e1f582f.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/8b61568d035315711602a4452864994b.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/76470a7016acb4c811279f5f1134cf45.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/ae1e594ef1a621071e996a43cc658ca5.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/f92d8566ff9460451ab42093098a0efe.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/6ef1d9e73ace130c61ae0817282d2f3e.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/11f9ae01877f6c0becf49c709eddb8cb.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/b4c1c35060baefabf07bd5b08b9d7886.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/1a9fab80d9712972c7381efd891b4688.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/d313f615dd44b07a6af2869f92665c41.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/627be9191d9dac23a05ba3d9776a03c2.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/906c2087a7215fbe7bd3f44adbf813a0.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/00360f79fd6e02781457eda48f85da90.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/0b8d4ebeeb72a935257f364d36619df1.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/1b861c23fc370c326ec8342733ef5d84.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/b30cedcfda7bc40f93bf95d7bffefa69.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/db0cdcca2a5a5cdeca7e1d107ce51eb2.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/5cd556902ceffbd521870f23d04610fb.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/1cb13f156bd436222447dd658180bd96.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/4cafa398999d41cc1bc82c114ca34b77.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/8f57485a1cf9daa97e41ec39b9b186b6.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/3775a16c58aed83497088371766a8fb4.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/8dd32e9c6b938487628f8da41928fcf2.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/d969c10c944af1a17e4c648b211295eb.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/e6268cdf7c7c51ceb6ee2dd76eb8d356.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/7e3b20fbc661625fa1c6a3c76975df51.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/4c2643b020ee1da6b95302657c243f87.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/8f03d97033b8047a49ad1042f4176aa9.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/af9c6a2eef8d8e112f368ce198e72885.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/6001ab14aa6ea1428cb3c67b64ba5711.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/d378871431ed1478bcc0f1d8b9120972.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/45cd53f9ee0ba8fef7b16b5aa061f960.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/e938d27a69495739699e989798ccfa5e.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/6f02497b7c4e1a5498cf886e15a9b254.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/d27178f5d1d6ac05f9bbe9dc0158d784.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/8b302b032be68618c85d4fffd32aae2b.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/383c188c66d24b641083e69a7e94941d.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/888577ffe6e13cb4e9c8e4d7b1e4b924.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/f64b4bcc2c8fcfdc3e2f603a6c2c7365.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/f43eaa14eb598d744ebbca0c3cce454b.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/adabe77c4e98d47595fcae9e0b8a6d78.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/23a0a0190b519b05aa748e6172728e15.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/db948894aa7e054cbd17a8298791f2a1.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/7bbcd6e1ed39753fe36adab9599be637.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/f05041385764c64376615154799b3c16.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/820f21f68db5a38bb0bbb9504ee783ac.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/b587a3c3dd78792865b2ec97de0b18f3.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/47e2a4a8e13ec7100f6af8cd839d1bb3.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/15a48116e4f978dd8c99ad1a6583a1b3.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/3cc3464e8cb309a47b799aa510b3fa3f.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/7235e754185d3321c4b6883d001a35ad.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/49c8336f2a448c245088a04a05247790.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/b54bd58453658571932fe8150f95855e.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/04b32b70b4ab15cad85d43e3b5359239.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/d0827541db38dafa0b462b70e2f5008f.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/6dc8a56fd5c5a503b8afe7aec63461b5.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/d79bd4ef79a6904f5269a673b148d650.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/23c57949a96da1057a7cd07516a9606c.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/87d5cdc85a4a22b73584c91049dae5f1.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/b8fc0bcbd1cc95db4d2887fa3c40ac5e.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/0322661ef29f9c81af295cf40f758469.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/623c761b4db398ea2157e6c5cd6c8c58.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/fae2676a3d4bd35b0b7088fad9f2e554.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/6017c62da21f2efd98a7f4d49470fc7e.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/3e4b3d6627946e46ed0c2f5be6dd2183.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/6bd29eb18f21ae52a50764c88dc22a46.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/758260238a2837b4acfaf02e05106c33.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/6cb8d88b1f9582bbe8191391cde1cebf.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/366b47c7a8ccab2cd0e5834dde998765.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/ef65be68dc196d4024ad499a4efd03fa.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/8a902f4a44a8c7d67e0ab012474ba00d.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/4847f9b1ba37f9970d647f419b43f768.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/b8c2ba8e0d4817153827fe65c426445e.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/8b78b25eecb0fc406d440de8efc0ff2c.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/1636ba745a6fc4d97dba1d27825de2b0.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/e9d689ab1d4af3dec583132948cc2273.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/61a6a8957de1bb20156d7a132715a155.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/0b2e8050a0c115b873563399e7c86ad6.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/ed6433fbbe878e362b4cf03d85418456.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/097dd567939b9ec200cbc29749fec135.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/4b9af890fe1cd191bba803ece97cce10.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/36c29b20a3e1ca94e4e15b63b8babe00.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/44d0e116c3fd7fc8b023876e8ad60ea6.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/fc19a5750ad5328368a5ef1d9775445b.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/101f6614e88de51d424770caa52669d9.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/ef8f036e4f9a914758e7216ec7ee71ea.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/751eeccb4219a04b7f3c6be293003054.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/43cf968c576544285cf11fb7f6c7b676.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/1603eddceceddc29f790fcf5ba04bec8.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/b414f08238785469af4ae86d24bdc75a.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/190e91556f1e3e63fc40938bcb27d8b2.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/5d4e4482ade1a29b502a3ad9f3a61cfc.a3d",
"../../../rec/data/PSRC/Data/stage1/a3d/65c89a7cccabe529cc81e0ab9ddea2ce.a3d"]
"""

# Load training and eval data
(train_data, train_labels, eval_data, eval_labels) = extract2DDataSet(image_path_list, block_size, segmentNumber, sc)
flattened_train = []
for item in train_data:
	flattened_train.append(item.flatten())
train_data = flattened_train

flattened_eval = []
for item in eval_data:
	flattened_eval.append(item.flatten())
eval_data = flattened_eval



###### Dimensionality Reduction ######

### PCA ###
"""
print("--- PCA Output ---")

print(len(eval_data + train_data))
pca = PCA(n_components=3)
pca.fit(eval_data + train_data)

print(pca.components_.shape)
print("Explained variance per component:")
print(pca.explained_variance_ratio_)  

# plot first (x) and second (y) components
plt.scatter(pca.components_[0], pca.components_[1])
plt.show()


###### Unsupervised Learning ######
"""
### K-Means ###
"""
print("--- K-Means Output ---")
kmeans_classifier = KMeans(n_clusters=2, random_state=0).fit(train_data)
print("Done training.")

predicted = kmeans_classifier.predict(eval_data)

# output results
print("Classification report for classifier %s:\n%s\n"
      % (kmeans_classifier, metrics.classification_report(eval_labels, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(eval_labels, predicted))

### AffinityPropagation ###
# WARNING: this is super memory intensive
print("--- AffinityPropagation Output ---")
affinity_classifier = AffinityPropagation(max_iter=50).fit(train_data)
print("Done training.")

predicted = affinity_classifier.predict(eval_data)

# output results
print("Classification report for classifier %s:\n%s\n"
      % (affinity_classifier, metrics.classification_report(eval_labels, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(eval_labels, predicted))

###### Supervised Learning ######
"""
### SVM ###
print("--- SVM Output ---")

# create the classifier
svm_classifier = svm.SVC(gamma=0.001)

# we learn on the training data
svm_classifier.fit(train_data, train_labels)
print("Done training.")

# now predict the threat
predicted = svm_classifier.predict(eval_data)

# output results
print("Classification report for classifier %s:\n%s\n"
      % (svm_classifier, metrics.classification_report(eval_labels, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(eval_labels, predicted))

### Basic Decision Tree ###
print("--- Decision Tree Output ---")

tree_classifier = tree.DecisionTreeClassifier()

#learn
tree_classifier = tree_classifier.fit(train_data, train_labels)
print("Done training.")

# predict
predicted = tree_classifier.predict(eval_data)

# output results
print("Classification report for classifier %s:\n%s\n"
      % (tree_classifier, metrics.classification_report(eval_labels, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(eval_labels, predicted))

### RandomForestClassifier ###

print("--- RandomForestClassifier Output ---")

forest_classifier = RandomForestClassifier(n_estimators=10)

#learn
forest_classifier = forest_classifier.fit(train_data, train_labels)
print("Done training.")

# predict
predicted = forest_classifier.predict(eval_data)

# output results
print("Classification report for classifier %s:\n%s\n"
      % (forest_classifier, metrics.classification_report(eval_labels, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(eval_labels, predicted))

"""
print("--- AdaBoostClassifier Output ---")

ada_classifier = AdaBoostClassifier(n_estimators=100)

#learn
ada_classifier = ada_classifier.fit(train_data, train_labels)
print("Done training.")

# predict
predicted = ada_classifier.predict(eval_data)

# output results
print("Classification report for classifier %s:\n%s\n"
      % (ada_classifier, metrics.classification_report(eval_labels, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(eval_labels, predicted))


### Gaussian Naive Bayes ###
print("--- Gaussian Naive Bayes Output ---")

gnb_classifier = naive_bayes.GaussianNB()

gnb_classifier = gnb_classifier.fit(train_data, train_labels)
print("Done training.")

predicted = gnb_classifier.predict(eval_data)

# output results
print("Classification report for classifier %s:\n%s\n"
      % (gnb_classifier, metrics.classification_report(eval_labels, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(eval_labels, predicted))

### Multinomial Naive Bayes ###
print("--- Multinomial Naive Bayes Output ---")

mnb_classifier = naive_bayes.MultinomialNB()

mnb_classifier = mnb_classifier.fit(train_data, train_labels)
print("Done training.")

predicted = mnb_classifier.predict(eval_data)

# output results
print("Classification report for classifier %s:\n%s\n"
      % (mnb_classifier, metrics.classification_report(eval_labels, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(eval_labels, predicted))


### Basic Multi-Layer Perceptron ###

from sklearn.preprocessing import StandardScaler  
scaler = StandardScaler()  
# Don't cheat - fit only on training data
scaler.fit(train_data)  
scaled_train = scaler.transform(train_data)  
# apply same transformation to test data
scaled_test = scaler.transform(eval_data)  

print("--- Basic Multi-Layer Perceptron Output ---")
from sklearn.neural_network import MLPClassifier


mlp_classifier = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)

mlp_classifier.fit(scaled_train, train_labels)
print("Done training.")

predicted = mlp_classifier.predict(scaled_test)

# output results
print("Classification report for classifier %s:\n%s\n"
      % (mlp_classifier, metrics.classification_report(eval_labels, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(eval_labels, predicted))

"""
