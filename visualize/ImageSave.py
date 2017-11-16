from classes import *
from dataExtraction import *
from random import shuffle
import matplotlib.pyplot as plt
import scipy.misc

# Save all images to a folder
class ImageSaver:
	def __init__(self, SupervisedClassifier):
		self.sc = SupervisedClassifier

	def save_images_to_folder(self, data = "2d", channels = 1, block_size = 40, resize = -1, segmentNumber=-100, image_filepath = "../../../rec/data/PSRC/Data/stage1/a3d/", nii_filepath = "data/Batch_2D_warp_labels/"):
		image_number = ""
		print ('beginning data accumulation')
		data_label_stream = []
		print ('looking for segment ', segmentNumber)

		image_path_list = self.get_image_set()
		print (len(image_path_list))

		real_block_size = block_size

		count = 0
		for image_path in image_path_list:
			count = count + 1
	        print(count, ': about to create a body scan with filepath ', image_path)
	        bs = BodyScan(image_filepath + image_path, nii_filepath)
	        bsg = BlockStreamGenerator(bs, self.sc, blockSize = block_size)
	        block_list = bsg.generate2DBlockStreamHandLabeled3Channel(saveImages=True, resize = resize)
	        
	        if(resize!=-1):
	            real_block_size = resize

	        for block in block_list:
	            if block[0].shape[0] == (real_block_size) and block[0].shape[1] == (real_block_size):
	                if(segmentNumber == block[1] or segmentNumber == -100):
	                    data_label_stream.append((block[0], int(block[2])))

	def create_train_test_arrays_from_images(save_path, X_train, Y_train, X_valid, Y_valid):
		np.save(save_path + "X_train_56_blocksize_3_channel", X_train)
		np.save(save_path + "Y_train_56_blocksize_3_channel", Y_train)
		np.save(save_path + "X_valid_56_blocksize_3_channel", X_valid)
		np.save(save_path + "Y_valid_56_blocksize_3_channel", Y_valid)

	def load_images_from_folder(max_folder, sum_folder, var_folder, resize):

	    max_image_filenames = os.listdir(max_folder)
	    sum_image_filenames = os.listdir(sum_folder)
	    var_image_filenames = os.listdir(var_folder)

	    training_length = len(max_image_filenames)/2
	    testing_lenth = len(max_image_filenames) - training_length

	    X_train = np.empty((training_length, resize, resize, 3))
	    Y_train = np.empty((training_length,2))
	    X_test = np.empty((testing_lenth, resize, resize, 3))
	    Y_test = np.empty((testing_lenth,2))

	    print 'here'
	    print len(max_image_filenames)
	    print max_image_filenames[10]

	    num_images = len(max_image_filenames)
	    for index in range(0,num_images):

	        print str(index) + "/" + str(len(max_image_filenames))
	        max_image_filename = max_image_filenames[index]
	        sum_image_filename = sum_image_filenames[index]
	        var_image_filename = var_image_filenames[index]

	        file_id, channel_type, is_threat, region, x, y = max_image_filename.split("_")
	        
	        # read in the image
	        max_array = scipy.misc.imread(os.path.join(max_folder,max_image_filename), mode = 'L')
	        sum_array = scipy.misc.imread(os.path.join(sum_folder,sum_image_filename), mode = 'L')
	        var_array = scipy.misc.imread(os.path.join(var_folder,var_image_filename), mode = 'L')

	        # resize the image
	        Channeled_Data = np.zeros((resize,resize,3))
	        data_channel_1 = scipy.misc.imresize(arr = max_array, size=(resize, resize))
	        data_channel_2 = scipy.misc.imresize(arr = sum_array, size=(resize, resize))
	        data_channel_3 = scipy.misc.imresize(arr = var_array, size=(resize, resize))

	        # add all the channels to the channeled data
	        for r in range(0,len(data_channel_1)):
	            for c in range(0,len(data_channel_1[0])):
	                Channeled_Data[r][c][0] = data_channel_1[r][c]
	                Channeled_Data[r][c][1] = data_channel_2[r][c]
	                Channeled_Data[r][c][2] = data_channel_3[r][c]

	        if(index < training_length):
	            X_train[index] = Channeled_Data
	            Y_train[index] = (1-int(is_threat),int(is_threat))
	        else:
	            X_test[index - training_length] = Channeled_Data
	            Y_test[index - training_length] = (1-int(is_threat),int(is_threat))

	    return X_train, Y_train, X_test, Y_test

	def get_image_set(self):
	        # Set image list for use - this is for my personal computer
	        #image_path_list = ["5e429137784baf5646211dcc8c16ca51.a3d"]
	        #return image_path_list

	        # full image list for server
	        image_path_list = ["6574d7241cad5f378da7dce9dfec4cd0.a3d",
	"8c70cc871902ae955d740cf1b7afc3e8.a3d",
	"87ab2075c257d92ec4bcb675b11d460f.a3d",
	"f6e4d412642e5cc4fcb0f6a08b592a04.a3d",
	"6f086abfb13481d705d0fb05cf6c06de.a3d",
	"7142e2ff6b927d5154afded4c90e2acd.a3d",
	"b4f8cdab7e00e33e5cfd233bc3965129.a3d",
	"832ea189a29a25ec10731796599231ba.a3d",
	"74244553005b05ece2981f71bcab30f7.a3d",
	"dc834b5d53aaab3b7cc909a101f3c263.a3d",
	"6745587060ddcad91da9d933d07719b2.a3d",
	"8ea3d183d3119e43f78b290889370d2a.a3d",
	"8d9fd0230cac0d2adfc1b7276fe1085e.a3d",
	"d65df1bd230816aa388018e37f239fd6.a3d",
	"172acab0b917b44f946c94051feab878.a3d",
	"acbd8470813a92fd5ff9c6d2c33ac614.a3d",
	"6d28992a920aac772121436b64196682.a3d",
	"1e4a14d2e1eb381b773446de1c0c0b7e.a3d",
	"ef54bd4acb418e3288009d97ea3d89d2.a3d",
	"12ed0db15024397106482c4983ad2d92.a3d",
	"5e327948c6493f2c4c8e5d1c6002fac4.a3d",
	"69cf1f6d3c1d52804542acea9f22a6b2.a3d",
	"0043db5e8c819bffc15261b1f1ac5e42.a3d",
	"16c8b78a7d69fb02bc1d74c8240c6171.a3d",
	"acd4b93d654f43b85d476c58ccfb8cf2.a3d",
	"71a28dc56c2039c74eba50a79831bfee.a3d",
	"e087226320cc189142228b5fb93ed58f.a3d",
	"6e5b2c089b7494dad05014132b1a38a0.a3d",
	"5cb8d7ef7176edcd0dd38e4c9921f185.a3d",
	"fc2e5de5fc3785fc09fd88f21286f75f.a3d",
	"4991020e3a552115864bd523dee8a353.a3d",
	"dcb8721ca9aca86afc1d7db6d3ff8ff3.a3d",
	"0fdad88d401b09d417ffbc490640d9e2.a3d",
	"775eaedd50c877e432de72a07c83ea79.a3d",
	"818281be1ffe4daa57fe8fc31b05b0d7.a3d",
	"f2c1d30f352f6b5ab8dd5da31f85ee1d.a3d",
	"b7eb90b7502b65755b0b063f94c8b428.a3d",
	"829295cbd850d4b74bdd9a3dfc539ba3.a3d",
	"e356161eeeadb1bbdeda6644e244e8fc.a3d",
	"366f12873367fcf4af617151a5f6daf5.a3d",
	"76470a7016acb4c811279f5f1134cf45.a3d",
	"81ece36076b9a707e5e4a0b25056b408.a3d",
	"89f3fc686b91265ce292639d61ac7dd5.a3d",
	"6ef1d9e73ace130c61ae0817282d2f3e.a3d",
	"239ca19250385147d033bd627592c582.a3d",
	"88f0b9b5d9c5e775a247eacc2f2948fc.a3d",
	"aec4d241a434f3d957c3fc7473db3135.a3d",
	"627be9191d9dac23a05ba3d9776a03c2.a3d",
	"906c2087a7215fbe7bd3f44adbf813a0.a3d",
	"00360f79fd6e02781457eda48f85da90.a3d",
	"83de41660e095242ddb6e1ff369dd710.a3d",
	"65881d8a53837de62f500ceba2d09623.a3d",
	"1cb13f156bd436222447dd658180bd96.a3d",
	"4cafa398999d41cc1bc82c114ca34b77.a3d",
	"2eb9eae48c9edc17c018b2ec10f18f94.a3d",
	"7828d838474e3b54306a164d1ba6419d.a3d",
	"8dd32e9c6b938487628f8da41928fcf2.a3d",
	"d969c10c944af1a17e4c648b211295eb.a3d",
	"123d65c19a4d7f395b3719d47d5666b5.a3d",
	"7e3b20fbc661625fa1c6a3c76975df51.a3d",
	"06f07ef12a24cd5ab6aa457ac8afa2a0.a3d",
	"4c2643b020ee1da6b95302657c243f87.a3d",
	"8f03d97033b8047a49ad1042f4176aa9.a3d",
	"af9c6a2eef8d8e112f368ce198e72885.a3d",
	"6001ab14aa6ea1428cb3c67b64ba5711.a3d",
	"84fa7e2bab404de5e5d513fda76cdf85.a3d",
	"afbc1bfc5487259b75f046aa02df9bdc.a3d",
	"e938d27a69495739699e989798ccfa5e.a3d",
	"6f02497b7c4e1a5498cf886e15a9b254.a3d",
	"b7fc53a239e7954a90ba61c2a139bb10.a3d",
	"d584337ad615d84929f144698c1f5efa.a3d",
	"7bbcd6e1ed39753fe36adab9599be637.a3d",
	"fbb1c4df6316910ca0e84ae0b5c2163e.a3d",
	"f876f77244d756c2dc2135fa1688577f.a3d",
	"15a48116e4f978dd8c99ad1a6583a1b3.a3d",
	"15f3ca9983fea662b727676941b8d378.a3d",
	"49c8336f2a448c245088a04a05247790.a3d",
	"b54bd58453658571932fe8150f95855e.a3d",
	"1e5f7a00ff4f02b0de0ea97139b6c92e.a3d",
	"87a11a5d8d1b59baefbcbd039422ead5.a3d",
	"fae2676a3d4bd35b0b7088fad9f2e554.a3d",
	"758260238a2837b4acfaf02e05106c33.a3d",
	"838462cf0fb5f9a44fed18dcccd61481.a3d",
	"6cb8d88b1f9582bbe8191391cde1cebf.a3d",
	"ef65be68dc196d4024ad499a4efd03fa.a3d",
	"7099aa697425acc504f85f5e9053761d.a3d",
	"f03b8634f43016d4a5059b5d5484a688.a3d",
	"68a9d05069d341a2b3da603a14c226a1.a3d",
	"f77d79b80819f351674f7d1bee0546de.a3d",
	"84bd8ac1e60b5fe6b5069dba094b75e1.a3d",
	"859e72ae9a5ca6c3d097a5eedf66b819.a3d",
	"61a6a8957de1bb20156d7a132715a155.a3d",
	"4c9aeaf8e9fca94a138529a06e02f27a.a3d",
	"0b2e8050a0c115b873563399e7c86ad6.a3d",
	"673f475cab0b37ea79ef7888e01f618c.a3d",
	"7a591b6ca71ad9f01eea3cfc369f0a94.a3d",
	"097dd567939b9ec200cbc29749fec135.a3d",
	"0a27d19c6ec397661b09f7d5998e0b14.a3d",
	"742084dcbc9f5d7dd863fb37a4be057a.a3d",
	"f19f248adb562842f69fe8255e50a3b6.a3d",
	"4967db4034491b1f8f06e726781c7d1c.a3d",
	"1680190be2fb589027ff8faadba65a9d.a3d",
	"1603eddceceddc29f790fcf5ba04bec8.a3d",
	"b414f08238785469af4ae86d24bdc75a.a3d",
	"65c89a7cccabe529cc81e0ab9ddea2ce.a3d",
	"12b3f3c81861b302d87117362b29a02e.a3d",
	"b5ba0c5a7e2e03c8e1a56e1dab8d4d87.a3d",
	"dfc053d81dd99368f5bd66d6d6f32710.a3d",
	"4b28d86db0cedb95323986f74e8fc894.a3d",
	"7e9a86c3a8979ef95ccce2073552cad6.a3d",
	"2eba5056e90c67e49c05991580cc7a4e.a3d",
	"8e1a7e8d24934e36e160985f32a8916b.a3d",
	"efcc0a2063eb1e9a101b57ad94345747.a3d",
	"5d6df35e4587f744cb39062fb56665ba.a3d",
	"895d15981c8e660962ac61d4544c7e6c.a3d",
	"faa3d6f358099ee2b091a5b87feca844.a3d",
	"7a465515247d5150a437499ed4dd31a8.a3d",
	"fb07ddc68370ee1a59ce61a50baf6ae3.a3d",
	"887b5406149d616263a90a0f56300a3d.a3d",
	"701383b75dcb508ae7310cbf944dc656.a3d",
	"6978eddfc19dde3a23059223b5ed3de8.a3d",
	"8ed75b644170d00153d4883854c559af.a3d",
	"37acb3827a8b9421866d55863974600a.a3d",
	"7312999be0cdddae2ac62adc96240970.a3d",
	"690e67cf1122bbe0a851f23498a38c39.a3d",
	"b44c9c3ab619b9ec06af18bb4847705d.a3d",
	"df788b51010c6de82fd1a3ff755ddf31.a3d",
	"67ba8c23c39914436baf3144e47a9a66.a3d",
	"32d505addd921c1f6fefb7ddc51e7f94.a3d",
	"904c11610724504ea061e04c1938fc00.a3d",
	"7be32cef8b55d2cfa63f76fb65b96032.a3d",
	"ec08e6f3d0e610914fcc26776554da85.a3d",
	"e299540987bf5ccf60ec2f2562ff67b1.a3d",
	"b92c30752529cf137775fabd79f13f58.a3d",
	"077c5701d8ee08e469ada0ef3b105cba.a3d",
	"6ff4d9c3196536a0c4935afd4535bd1e.a3d",
	"38fdbc727d99dcf6f22371b5b7c96992.a3d",
	"412d88bd9a4612b71759fa802a27f19e.a3d",
	"e9e2bc4f4f319943935551d502fc11c2.a3d",
	"43d1a815858d7584eb6e0e84f8a1bf8b.a3d",
	"ec9c7903d4665303f7d3150399af8d84.a3d",
	"8c49540486913f57c2c543a563d9b9a7.a3d",
	"471829836c7df1fa0c63721d09ea6db9.a3d",
	"6c4acb4d8cc81e0ce4f18a3cb66a9791.a3d",
	"e2b431b503c61a1044815b9c53f3fe4f.a3d",
	"fd3111f15da4a2052b2b50bfc5ef6465.a3d",
	"e38dcb351fe15acd74d1b0352c0e0e48.a3d",
	"8db2645372fe8c4fcb62319a70deb914.a3d",
	"e195068859c5847b028c905dd1ccb81d.a3d",
	"b53eee6d51543953cdc99756df81e3fb.a3d",
	"1f0ecd9585289e8848549575223caaeb.a3d",
	"0240c8f1e89e855dcd8f1fa6b1e2b944.a3d",
	"08daf3a6cdb4f5e1ad15cb2431ea419a.a3d",
	"416a6888eb7ad8c4416fe0b620435136.a3d",
	"fb397be3a8a1e9312d8b97a95bc97ae6.a3d",
	"de61784ab35622c44df1858da3252f6a.a3d",
	"ea729b325bbf9762d2d31ba6dbae62b3.a3d",
	"e9c97bb2cf608b1f0e23f078b10502bc.a3d",
	"13bc14ee7bc8d31a150b0744a6f1e0fe.a3d",
	"ef7bca7848bdf159c963a2a40bf50a09.a3d",
	"adf2e22d579cbfb345585ca757c4db58.a3d",
	"f58f9f63785b5e7c996cacab2e1f582f.a3d",
	"fc6fc6d5f3f714fbac11dd1491ea2b29.a3d",
	"467cfa7996fcdb7e4f5c4d64c2a01de2.a3d",
	"b130ae39a79d539b350b0a68c9a9d28d.a3d",
	"3ea9575fe8926bbb01e02b77c0802668.a3d",
	"7ceae9953c62b3337c12712ac53917b7.a3d",
	"0d10b14405f0443be67a75554da778a0.a3d",
	"35fcc4eea20865907dd58ad98684d52b.a3d",
	"d313f615dd44b07a6af2869f92665c41.a3d",
	"b433a0a67ad4efa50d5f65c5d8eb69e3.a3d",
	"0b8d4ebeeb72a935257f364d36619df1.a3d",
	"13c13b32a6f3818169394c16eae3ea52.a3d",
	"db0cdcca2a5a5cdeca7e1d107ce51eb2.a3d",
	"1eb2501ae493867f637a9a3c08d4c17c.a3d",
	"d8710a2be4fa28baec042736d9537ebf.a3d",
	"d378871431ed1478bcc0f1d8b9120972.a3d",
	"8a287749ed34fd0d2ba168e8643cdd73.a3d",
	"8c9996e3af3394e87179bf2b29e2400a.a3d",
	"eac3d5c446fc041486d325a648a1e72c.a3d",
	"45cd53f9ee0ba8fef7b16b5aa061f960.a3d",
	"afd4a630b3848ffd5032941f231fed39.a3d",
	"41ed36bdc93a2ff5519c76f263ab1a88.a3d",
	"b6c2b509637bd5d1cf58bc0387265967.a3d",
	"f64b4bcc2c8fcfdc3e2f603a6c2c7365.a3d",
	"8438a4ad2a1dbb3a79d6595349390e8f.a3d",
	"f43eaa14eb598d744ebbca0c3cce454b.a3d",
	"3eda71e0fd6f0c18c1fa43371c4212e4.a3d",
	"19880a67d084ab5a0c3598051cef28e5.a3d",
	"f05041385764c64376615154799b3c16.a3d",
	"674a33b5051197c820f44d9b7e441dd7.a3d",
	"47e2a4a8e13ec7100f6af8cd839d1bb3.a3d",
	"8694613a993813b497573680e7a2ef04.a3d",
	"3cc3464e8cb309a47b799aa510b3fa3f.a3d",
	"87114abe7ec898d88828d5a74929534f.a3d",
	"ec892048fb389584eac857af4edfc2bd.a3d",
	"4869f57f719b26a01dc7d196abe4f73d.a3d",
	"8ef460e13de465fc4c3c12292be62904.a3d",
	"e972396f2c525bfc18bfdb300b3510d1.a3d",
	"e3169583747cf8cceac6b06925703101.a3d",
	"ea6e3b358c16a95112e71f504c136dfe.a3d",
	"86753997b29547ca352db8a3c35b0749.a3d",
	"04b32b70b4ab15cad85d43e3b5359239.a3d",
	"8411eea809837acfdf4bb282d0e5db68.a3d",
	"b841d8010bae6bd1a14e9f36125892f5.a3d",
	"8eebb6632b702f343740a17a242d9aaa.a3d",
	"3e4b3d6627946e46ed0c2f5be6dd2183.a3d",
	"5db153532b62bba69d8076aad7119e57.a3d",
	"f952090eaf6f6e1ebd502f0ce029e862.a3d",
	"2fc4b9b35bf2e764c9ce32c66646fa04.a3d",
	"0d06d6cafd95360fb15573181f8f3d9a.a3d",
	"2fe50818cf314d93b93043f492bd13ce.a3d",
	"8a902f4a44a8c7d67e0ab012474ba00d.a3d",
	"69d59a4f636cf8377a2d1d9d117d6310.a3d",
	"e9d689ab1d4af3dec583132948cc2273.a3d",
	"36c29b20a3e1ca94e4e15b63b8babe00.a3d",
	"101f6614e88de51d424770caa52669d9.a3d",
	"ef8f036e4f9a914758e7216ec7ee71ea.a3d",
	"630db444dfa39d87671dce7e4163a009.a3d",
	"47da5b8b67428fbdcfacdac9f77a8dfc.a3d",
	"b93f543e66317a22645be549dc1c008a.a3d",
	"e253adde4a0a6733cbf12a8e8a17d2d0.a3d",
	"e86230438a2d639091e811f7145c263c.a3d",
	"3a8696b99b2d1b28be62389d48d697be.a3d",
	"0c64d2701479ab8733a412187338da75.a3d",
	"72ac8e35980553af85d88437fbe2269f.a3d",
	"006ec59fa59dd80a64c85347eef810c7.a3d",
	"137cd2942a4d022921fda492ff79d40f.a3d",
	"4374545e0e29d0c17cacb7d4453498b0.a3d",
	"0ada538288f0b62b01510b397a8acb9e.a3d",
	"413c6337a5b11c91f24c648c165b3f1a.a3d",
	"fbd3ca2d2af1785bfe3af2629f35bc3f.a3d",
	"e8fcca2e33a61919fb9abf405fa73c1e.a3d",
	"d798c87a5b4ad5411079766152aaaddc.a3d",
	"1beb3744a2da2da340ef7b7f2b793e32.a3d",
	"ae783a8040b1c8c0e729e787097e4113.a3d",
	"ee58b1e752416fbd0c0928d1a49a13a6.a3d",
	"3b46b70b40bb4f77a9da529f06356b50.a3d",
	"7d8935f1992f7df23cb5d1cf2c987a63.a3d",
	"8a3a0461a76394d6d2ae82cc34cb11c5.a3d",
	"6169b4632782ed2700716df0be91a860.a3d",
	"8464ffead51e673a5244a1b5c0c63b86.a3d",
	"fd23dddfcd06dcb4dacc1731a348b8cd.a3d",
	"e11196e89d40e8e964e9bfcd989703b3.a3d",
	"435cc314ea7e77733c95bcf05d252259.a3d",
	"ebd66ad6f4f712e118714fdd9bbc7846.a3d",
	"e330c71147b29b83782fb79b219efa0f.a3d",
	"84312f6fe146f36c0da50a60cb4b3b70.a3d",
	"4881eeb549b78b3fe90f9cdde9a5f649.a3d",
	"e00e3a2b4b32513eb31e203e1bb91d87.a3d",
	"88de0795410e7fe94d1c86868a3ca053.a3d",
	"499aa07eb418b3d46209d5f937d45d67.a3d",
	"86d97d6bb6bbfd6542954ebbdb2ff28e.a3d",
	"011516ab0eca7cad7f5257672ddde70e.a3d",
	"17281b53b9a17b1b5fe9ffefac82289f.a3d",
	"8085d6bf5256a67fd53b545dbb53caf4.a3d",
	"f6f0834b756940c73a4424cedeaee3b1.a3d",
	"6d0e25601dc8d96e71c7323d1dec6903.a3d",
	"adce87ff23a76a71cf379950008ab2a2.a3d",
	"edfaf43f833e92abb754d6e37b6bb2e9.a3d",
	"48f1d872def62e5d5150c1a7d64d1196.a3d",
	"4826cf5a1a8b1958ef4a9b00c941be58.a3d",
	"690c0567ddd681f71652841feab25cea.a3d",
	"f35a31e8b666ba97841c98ae6a26f3ef.a3d",
	"da92d0aae2f5450422d6306d450b90a0.a3d",
	"401dc40c30b9206019d61c4ed625cfce.a3d",
	"0e34d284cb190a79e3316d1926061bc3.a3d",
	"fdb996a779e5d65d043eaa160ec2f09f.a3d",
	"b41ded8fb8ebac447cbe0c50574fc9ed.a3d",
	"eede02f9558ebf3c51f502b843826498.a3d",
	"8b61568d035315711602a4452864994b.a3d",
	"ae1e594ef1a621071e996a43cc658ca5.a3d",
	"f92d8566ff9460451ab42093098a0efe.a3d",
	"f49798fea19cafb93d31109d7bfce948.a3d",
	"73135e2ad319f4f2cfe8d60bc6360759.a3d",
	"f7a768df99f624a0784950a2f7a965e0.a3d",
	"b4c1c35060baefabf07bd5b08b9d7886.a3d",
	"f5283ffe7e484ffc640ebcf62b534f7b.a3d",
	"b77b5c4337e422dcbc29be19d2c1d355.a3d",
	"1a9fab80d9712972c7381efd891b4688.a3d",
	"827f4ce527c4da872a09067901d7ec72.a3d",
	"aee24ed64ef8b1d7ab5e24c551478f9a.a3d",
	"45ac0b88f67663a1d30dcfc8366fac31.a3d",
	"0d925b71485ba1f293ef8abb53fcd141.a3d",
	"06ff02c1f431d0c21b945b4ea459cc88.a3d",
	"fcd7df5b42fcd420d03ac7bdec718195.a3d",
	"1b861c23fc370c326ec8342733ef5d84.a3d",
	"f96fe81c61c951792a43cb4851ca3829.a3d",
	"5cd556902ceffbd521870f23d04610fb.a3d",
	"d727f5bae0dac8319fd5c653b1c4b898.a3d",
	"8f57485a1cf9daa97e41ec39b9b186b6.a3d",
	"fd313303746277a1358447f624762dcd.a3d",
	"1784b1d51fdfb02d8ba0686255287e76.a3d",
	"f859802d4f5cec2b3220725ce54b3322.a3d",
	"3f1c3df84fcc2a555d1a060ae1d24ed7.a3d",
	"3dd66b02f0817f039dd0619aa1a7ef13.a3d",
	"e6268cdf7c7c51ceb6ee2dd76eb8d356.a3d",
	"19288e1c3e8faf7f3948a4e8f1848ae9.a3d",
	"8697dc7ab1bb4ec709ff3daed6e84c3d.a3d",
	"d27178f5d1d6ac05f9bbe9dc0158d784.a3d",
	"098f5cfcf6faefd3011a94719cb03dc5.a3d",
	"8ff23c0d13a4bc785fab433979b8621a.a3d",
	"383c188c66d24b641083e69a7e94941d.a3d",
	"7724e79d4e24c775156d55db337ac6d6.a3d",
	"adabe77c4e98d47595fcae9e0b8a6d78.a3d",
	"23a0a0190b519b05aa748e6172728e15.a3d",
	"0535d52b89f5dcca44563b62592ca366.a3d",
	"678074e9d3aa9ab9fec3f348002f5ca0.a3d",
	"76a01e339a2d8833d072ba4181409f28.a3d",
	"820f21f68db5a38bb0bbb9504ee783ac.a3d",
	"45073233709552ca8bc433e39b469b69.a3d",
	"6499c1b7896abcc72bb887a7ae717ba1.a3d",
	"865895a508b0feaa8b136b0eb747668a.a3d",
	"452d00e6fb36895567f246b9df3bd947.a3d",
	"8f25f226025e038b164941cb3176ea68.a3d",
	"1d6dd266503bccaae7b8ab100f6fac97.a3d",
	"ee08cba6fcdc5738a4ebcc8f6d57de03.a3d",
	"d0827541db38dafa0b462b70e2f5008f.a3d",
	"6dc8a56fd5c5a503b8afe7aec63461b5.a3d",
	"d79bd4ef79a6904f5269a673b148d650.a3d",
	"7c1f17a0060a103e5f3363b36780407b.a3d",
	"e90b05c111e7f0899d32bd2bad832961.a3d",
	"da0fb27c406b931e7199c5d087f8ef82.a3d",
	"87d5cdc85a4a22b73584c91049dae5f1.a3d",
	"0322661ef29f9c81af295cf40f758469.a3d",
	"623c761b4db398ea2157e6c5cd6c8c58.a3d",
	"82d9c305d2a2895c900f912f661a7464.a3d",
	"052117021fc1396db6bae78ffe923ee4.a3d",
	"6bd29eb18f21ae52a50764c88dc22a46.a3d",
	"7b698050fe145ce976a67d04058212d3.a3d",
	"8997400063d9cf8537b786a231aa08e0.a3d",
	"76ce1149faeced69afa3dc256869d0d8.a3d",
	"4847f9b1ba37f9970d647f419b43f768.a3d",
	"b8c2ba8e0d4817153827fe65c426445e.a3d",
	"8b78b25eecb0fc406d440de8efc0ff2c.a3d",
	"4b9af890fe1cd191bba803ece97cce10.a3d",
	"44d0e116c3fd7fc8b023876e8ad60ea6.a3d",
	"fc19a5750ad5328368a5ef1d9775445b.a3d",
	"5c1c4e08f2ad592d7dc90f1d7c5be3df.a3d",
	"0097503ee9fa0606559c56458b281a08.a3d",
	"0748878870d90c426ee75d30aed5b024.a3d",
	"6bfefb348d746eca288c6d62f6ebec04.a3d",
	"43cf968c576544285cf11fb7f6c7b676.a3d",
	"8b267d4997155cc9b502103371db9c44.a3d",
	"190e91556f1e3e63fc40938bcb27d8b2.a3d",
	"5d4e4482ade1a29b502a3ad9f3a61cfc.a3d",
	"d73f48c83070cd90ea0c024b93951979.a3d",
	"e5133432b54a778fb14f43c72e9e9124.a3d",
	"831600b1b6984119fc87529bf4b61ade.a3d",
	"5cbbcb73ae6e7fa727d41b2787d91d6a.a3d",
	"db73cda1eecf5c53168b9e60dbfe4600.a3d",
	"b2aef1eb2d498341049cd1ca6c99e06e.a3d",
	"14f44a8d741ff33faccfc18f7cde6668.a3d",
	"b60c5e2b4e6faafccbdc2e6893428481.a3d",
	"ebf9858fc6af11b1d3e66dc2a5201bf1.a3d",
	"1a10297e6c3101af33003e6c4f846f47.a3d",
	"8d5861df3b346dbc595694a39152b4fd.a3d",
	"31607601930f54ef5d7dcc4dd11ee822.a3d",
	"87355c6d2c2b21543bbc8e9d0c3c6a87.a3d",
	"e78e46f1dc5490ffa8925bc6c277b560.a3d",
	"3e5e4ede1b6a4ec767480b261336b1b5.a3d",
	"b220ad927a2a32a1e5bd329c4960a443.a3d",
	"1f4b484a158b1af3c3ae19e10406647e.a3d",
	"3c0668db35915783be0af87b9fa53317.a3d",
	"e1015cda16ab447880b50054bfa6fac4.a3d",
	"1c2b0a83d46ae57fb806306a444e165b.a3d",
	"5e429137784baf5646211dcc8c16ca51.a3d",
	"b02f6791d2b027a042de141562fc345a.a3d",
	"f6aa79cb1b8f487ab0dfd0c9a3943b50.a3d",
	"826b3b5eb25ddd6f7d2aed1e531e69b9.a3d",
	"b07aab4469700f4937e71e617637e0b9.a3d",
	"b5a09d3568108559bae352404708e1c9.a3d",
	"6307c2e0b60d78fd104b90b0f26ad0df.a3d",
	"44ac0d2d076928590d80acfda3871de8.a3d",
	"fa0eddae7fa3969578d8216fe1840e4c.a3d",
	"d360f28e70e6ec1aabb8a2ef5237491d.a3d",
	"e06b9551603d329dfe720f67e37ce87b.a3d",
	"eb3e1410bfa1f034f6bed038684cfc29.a3d",
	"84e2375926811ef6ca836e2bd172997f.a3d",
	"333c64b18d7e64f0d53bf73cea1ab5ec.a3d",
	"4b19940919dffa8c6611451e650e7c11.a3d",
	"d78bcc549aaa09af770f2a05fbc97e5b.a3d",
	"82c88e8b9f2c84cbfb9c9ab0392e166d.a3d",
	"43104359ffec7bd07385479d840ef5b5.a3d",
	"4275a47cbd350f67c9691d7b52313ac1.a3d",
	"e466360d4bb93b710f63518eab8b37b9.a3d",
	"01941f33fd090ae5df8c95992c027862.a3d",
	"48309fc9d12a28f0ba94b8c6d81bc3c0.a3d",
	"e75ec32aa94a4d092e6f59c3194aff2a.a3d",
	"d390855c1cfd9c19317341a315bcc4bd.a3d",
	"11f9ae01877f6c0becf49c709eddb8cb.a3d",
	"d0673b67054fa70934be1775df4c902f.a3d",
	"2f5c066720d997f33453dc491141bc70.a3d",
	"eb358f0ab6a6175f8a9617f1e5530d6d.a3d",
	"01c08047f617de893bef104fb309203a.a3d",
	"42181583618ce4bbfbc0c4c300108bf5.a3d",
	"b30cedcfda7bc40f93bf95d7bffefa69.a3d",
	"fcf366a672d2809b0d158180ecb4dd5c.a3d",
	"3775a16c58aed83497088371766a8fb4.a3d",
	"fe350eda846477990a05822cc76a2219.a3d",
	"822f26a77eaca1f06fcda124a494710e.a3d",
	"3b95713b5eddae2228e1fabe96e54678.a3d",
	"69827aa4c8beb31f0ca90c207d719085.a3d",
	"8b302b032be68618c85d4fffd32aae2b.a3d",
	"888577ffe6e13cb4e9c8e4d7b1e4b924.a3d",
	"e984c01d47f95816b214ea29ff943660.a3d",
	"db948894aa7e054cbd17a8298791f2a1.a3d",
	"858e731e41bb5cc224b26326376c86f1.a3d",
	"e4169734e5c2599df873d5e1bb019766.a3d",
	"06726105fabadca043737601d06415a8.a3d",
	"71e4417b386a04fdecef5f2888ef5805.a3d",
	"b587a3c3dd78792865b2ec97de0b18f3.a3d",
	"8e207447210b0e85cf086ed5963dfdce.a3d",
	"0fc066d8ab1c5a6a42b636c1fc5876a6.a3d",
	"66785ad57aecc8659e794371bb0d61af.a3d",
	"f0e2e0da0d391e1d34628e26080c89ac.a3d",
	"7235e754185d3321c4b6883d001a35ad.a3d",
	"8ade614868e3c0deed9a7e5d30927810.a3d",
	"130e2a40398af1cd77de5935b913f577.a3d",
	"84500d50c50b18be85decfe6751017c7.a3d",
	"7b47e17a89dc465f3882aa2a14299bf5.a3d",
	"fa33b3e3464e9c745cd93737c69e6c52.a3d",
	"742c1108ff62fac7b656c25e10ff2e32.a3d",
	"23c57949a96da1057a7cd07516a9606c.a3d",
	"628cfb55389acec85fffdfc520f3684b.a3d",
	"b8fc0bcbd1cc95db4d2887fa3c40ac5e.a3d",
	"6017c62da21f2efd98a7f4d49470fc7e.a3d",
	"1835b242ed6c519cc0017cdd5b08aa9b.a3d",
	"40e23061cdbfebd302a34f8ced45636e.a3d",
	"f1d5226e5afc63f5a696b7b1b64ae751.a3d",
	"135089e54a97b70ab3dce4f3dd4684da.a3d",
	"366b47c7a8ccab2cd0e5834dde998765.a3d",
	"f5610c13360bc50bd938899b41004fb6.a3d",
	"1636ba745a6fc4d97dba1d27825de2b0.a3d",
	"ed6433fbbe878e362b4cf03d85418456.a3d",
	"7937949be388818d91766d2566f0e8cf.a3d",
	"63781f8e65112e3f7bdc54394abc2310.a3d",
	"0958be20b77ecd4b4e19d749a271b7bb.a3d",
	"751eeccb4219a04b7f3c6be293003054.a3d",
	"ede2eaf52e9d66850db0466c855d889e.a3d",
	"e69ebd61a31b52c1ff560477534cf80b.a3d",
	"3f276a1a6cb52c371bad9a651e8e9bbf.a3d",
	"6d28ec3d68a372419e582cfb6f78c97f.a3d"]
	        return image_path_list 

class DataSetTSA:
    def __init__(self, trD, trL, teD, teL):
        self.trainingData = trD
        self.trainingLabels = trL
        self.testingData = teD
        self.testingLabels = teL

    def getTrainingData(self):
        return self.trainingData
    def getTrainingLabels(self):
        return self.trainingLabels
    def getTestingData(self):
        return self.testingData
    def getTestingLabels(self):
        return self.testingLabels
