from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
import numpy as np

# See https://keras.io/api/applications/ for details

vgg_linefeatrue = False
resnet_blockfeature = True
show_layers

class FeatureExtractor:
    def __init__(self):
    	if vgg_linefeatrue:
		#base_model = VGG16(weights='imagenet')
		#self.model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)

	if resnet_blockfeature:
		base_model = ResNet50(weights='imagenet')
		#self.model = Model(inputs=base_model.input, outputs=base_model.get_layer('conv4_block6_out').output)  # 28x28
		
		self.model = Model(inputs=base_model.input, outputs=[
									base_model.get_layer('conv3_block4_out').output,
									base_model.get_layer('conv4_block6_out').output
									])  # 28x28
	
	if show_layers:	
		layers = base_model.layers
		for l in layers()
			print(l.name)
	

    def extract(self, img):
        """
        Extract a deep feature from an input image
        Args:
            img: from PIL.Image.open(path) or tensorflow.keras.preprocessing.image.load_img(path)

        Returns:
            feature (np.ndarray): deep feature with the shape=(4096, )
        """
        #img = Image.open(img_path))
        img = Image.fromarray(img)
        img = img.resize((224, 224))  # VGG must take a 224x224 img as an input
        img = img.convert('RGB')  # Make sure img is color
        x = image.img_to_array(img)  # To np.array. Height x Width x Channel. dtype=float32
        x = np.expand_dims(x, axis=0)  # (H, W, C)->(1, H, W, C), where the first elem is the number of img
        x = preprocess_input(x)  # Subtracting avg values for each pixel
        
        with tf.device("/gpu:0")
		feature = self.model.predict(x)[0]  # (1, 4096) -> (4096, )
		print(feature.shape)
		return feature / np.linalg.norm(feature)  # Normalize
        
        
if __name__ == "__main__"
	fe = FeatureExtractor()	
	feature1 = fe.extract(img1)
	feature2 = fe.extract(img2)
	dists = np.linalg.norm(features-query, axis=1) 
	print("distance", dists)
	
	
	
